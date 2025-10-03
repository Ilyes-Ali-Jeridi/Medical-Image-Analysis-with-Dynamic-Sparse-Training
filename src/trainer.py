#trainer.py
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
import logging
from typing import Dict, Any, Tuple, List
import wandb

from .dataset import MIMICCXRDataset
from .models import MedicalRAG, MedicalViT, DynamicSparseLayer
from .optimizer import A100Optimizer

logger = logging.getLogger(__name__)

class DistributedRadiologyTrainer:
    """
    Handles distributed training of radiology report generation models.

    This class orchestrates the training process, including setting up
    distributed environments, preparing data loaders, managing the model
    (including DDP wrapping), optimizer, and training loop. It also integrates
    with Weights & Biases (wandb) for experiment tracking and handles
    checkpointing. Automatic Mixed Precision (AMP) is utilized via GradScaler.

    Attributes:
        config (Any): Configuration object containing hyperparameters and settings.
        device (torch.device): The primary device for this process (e.g., 'cuda:0').
        model (torch.nn.Module | DDP): The core model, potentially wrapped in DistributedDataParallel.
        optimizer (torch.optim.Optimizer): The optimizer for model parameters.
        scaler (torch.cuda.amp.GradScaler): GradScaler for AMP.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
    """
    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        config: Any, # Should ideally be a specific Config class instance
        device: Optional[torch.device] = None
    ):
        """
        Initializes the DistributedRadiologyTrainer.

        Args:
            train_csv (str): Filename of the training data CSV (relative to config.data_dir).
            val_csv (str): Filename of the validation data CSV (relative to config.data_dir).
            config (Any): A configuration object with all necessary training parameters
                          (e.g., learning_rate, batch_size, model params, paths).
            device (Optional[torch.device], optional): The PyTorch device to use. 
                                                       If None, it's auto-configured. Defaults to None.
        
        Raises:
            Exception: Propagates exceptions from model/optimizer/dataloader setup if critical.
        """
        logger.info("Initializing DistributedRadiologyTrainer...")
        self.config = config
        # Configure device using A100Optimizer or use provided device
        self.device = device or A100Optimizer.configure_device() 
        
        logger.info("Setting up distributed training environment...")
        self.setup_distributed() # Configures torch.distributed if necessary
        
        logger.info("Initializing model components (MedicalViT, MedicalRAG)...")
        try:
            # Initialize vision encoder using parameters from config
            vision_encoder = MedicalViT(
                sparse_rate=self.config.sparse_rate,
                img_size=self.config.image_size,
                patch_size=self.config.patch_size,
                embed_dim=self.config.embed_dim,
                num_layers=self.config.num_layers
                # num_heads can be added to config if needed from self.config
            )
            # Initialize the main RAG model with the vision encoder
            self.model = MedicalRAG(vision_encoder=vision_encoder, sparse_rate=self.config.sparse_rate)
            self.model.to(self.device) # Move model to the designated device
            logger.info(f"Model initialized successfully and moved to device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}", exc_info=True)
            raise # Critical failure

        # Wrap model with DistributedDataParallel (DDP) if in distributed mode
        if dist.is_initialized():
            logger.info(f"Wrapping model with DistributedDataParallel for rank {dist.get_rank()}.")
            self.model = DDP(
                self.model,
                device_ids=[dist.get_rank()],       # Current process's GPU
                output_device=dist.get_rank(),      # Where to gather outputs
                find_unused_parameters=getattr(self.config, 'ddp_find_unused_parameters', False) # Configurable
            )
            logger.info("Model successfully wrapped with DDP.")
        
        logger.info("Initializing AdamW optimizer...")
        try:
            # Use dynamically calculated learning rate from config if available
            lr_to_use = getattr(self.config, 'lr', self.config.base_lr) # Fallback to base_lr if 'lr' not set
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), # Get parameters from model (DDP handles .module)
                lr=lr_to_use, 
                weight_decay=self.config.weight_decay
            )
            logger.info(f"Optimizer initialized with LR: {lr_to_use:.2e}, Weight Decay: {self.config.weight_decay:.2e}")
        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {e}", exc_info=True)
            raise # Critical failure
        
        # Initialize GradScaler for Automatic Mixed Precision (AMP) training
        logger.info("Initializing GradScaler for Automatic Mixed Precision (AMP).")
        self.scaler = GradScaler(enabled=getattr(self.config, 'use_amp', True)) # AMP enabled by default, can be configured
        
        logger.info("Setting up data loaders...")
        self.setup_data_loaders(train_csv, val_csv) # This method logs its own progress
        
        # Initialize Weights & Biases (wandb) for experiment tracking on the master process
        if self.is_master():
            logger.info("Master process: Initializing Weights & Biases (wandb)...")
            try:
                # Pass relevant parts of config to wandb, or all of it
                wandb.init(project=getattr(self.config, 'wandb_project', "medical-rag-training"), 
                           config=vars(self.config), # Log all config parameters
                           name=getattr(self.config, 'run_name', None)) # Optional run name
                logger.info("Wandb initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize wandb: {e}. Training will continue without wandb.", exc_info=True)
                # Non-critical, so training can proceed.
    
    def setup_distributed(self):
        """
        Sets up the distributed training environment using `A100Optimizer`.
        This is called during `__init__`.
        """
        if not dist.is_initialized():
            logger.info("Distributed environment not initialized by external launcher. Configuring now via A100Optimizer.")
            A100Optimizer.configure_distributed() # This method handles its own detailed logging
        else:
            logger.info("Distributed environment was already initialized (e.g., by torchrun/slurm).")
    
    def setup_data_loaders(self, train_csv: str, val_csv: str):
        """
        Initializes training and validation DataLoaders.

        Creates `MIMICCXRDataset` instances for training and validation,
        and wraps them in `DataLoader` with appropriate samplers (DistributedSampler
        if in distributed mode).

        Args:
            train_csv (str): Filename of the training CSV.
            val_csv (str): Filename of the validation CSV.
        
        Raises:
            FileNotFoundError: If CSV files are not found.
            Exception: For other errors during dataset/dataloader creation.
        """
        logger.info(f"Setting up training dataset from: {self.config.data_dir / train_csv}")
        try:
            # Create training dataset instance
            train_dataset = MIMICCXRDataset(
                csv_file=train_csv,
                data_dir=self.config.data_dir,
                cache_dir=self.config.cache_dir,
                transform=getattr(self.config, 'train_transform', None) # Allow custom transform from config
            )
            logger.info(f"Training dataset loaded: {len(train_dataset)} samples.")
        except FileNotFoundError as e:
            logger.error(f"Training CSV file not found at {self.config.data_dir / train_csv}: {e}", exc_info=True)
            raise
        except Exception as e: # Catch other dataset instantiation errors
            logger.error(f"Error creating training dataset: {e}", exc_info=True)
            raise

        logger.info(f"Setting up validation dataset from: {self.config.data_dir / val_csv}")
        try:
            # Create validation dataset instance
            val_dataset = MIMICCXRDataset(
                csv_file=val_csv,
                data_dir=self.config.data_dir,
                cache_dir=self.config.cache_dir,
                transform=getattr(self.config, 'val_transform', None) # Allow custom transform from config
            )
            logger.info(f"Validation dataset loaded: {len(val_dataset)} samples.")
        except FileNotFoundError as e:
            logger.error(f"Validation CSV file not found at {self.config.data_dir / val_csv}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error creating validation dataset: {e}", exc_info=True)
            raise
        
        # Create samplers: DistributedSampler for DDP, None for single GPU/CPU
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=getattr(self.config, 'seed', 42)) if dist.is_initialized() else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.is_initialized() else None
        
        logger.info(f"Creating DataLoader for training data. Batch size: {self.config.batch_size}, Num workers: {self.config.num_workers}, Distributed: {dist.is_initialized()}")
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size_per_gpu if dist.is_initialized() else self.config.batch_size, # Adjust batch size per GPU
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True, # Improves data transfer speed to GPU
            drop_last=True   # Ensures all batches have the same size, useful for some models/DDP
        )
        
        logger.info(f"Creating DataLoader for validation data. Batch size: {self.config.batch_size}, Num workers: {self.config.num_workers}, Distributed: {dist.is_initialized()}")
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size_per_gpu if dist.is_initialized() else self.config.batch_size,
            sampler=val_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False # Evaluate on all validation samples
        )
        logger.info("Data loaders created successfully.")
    
    @staticmethod
    def is_master() -> bool:
        """
        Checks if the current process is the master process in a distributed setup.
        In non-distributed setup, always returns True.
        """
        return not dist.is_initialized() or dist.get_rank() == 0
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, List[str]]:
        """
        Performs a single training step (forward pass, loss calculation, backward pass, optimizer step).

        Uses Automatic Mixed Precision (AMP) via `torch.cuda.amp.autocast` and `GradScaler`.

        Args:
            batch (Dict[str, torch.Tensor]): A batch of data from the DataLoader, containing
                                             'images', 'labels' (for classification loss), 
                                             and 'report' (text for generation loss).

        Returns:
            Tuple[float, List[str]]: A tuple containing the loss value for the step (float)
                                     and a list of generated report strings from this step.
        
        Raises:
            RuntimeError: If a CUDA OOM error or other critical runtime error occurs.
            Exception: For other unexpected errors during the training step.
        """
        self.model.train() # Set model to training mode
        self.optimizer.zero_grad(set_to_none=True) # More memory efficient way to zero gradients
        
        # Move data to the configured device (non_blocking for potential overlap)
        images = batch['images'].to(self.device, non_blocking=True)
        labels = batch['labels'].to(self.device, non_blocking=True) # For classification part of loss
        reports = batch['report'] # List of strings, stays on CPU, handled by tokenizer in model
        
        try:
            # Automatic Mixed Precision context
            with autocast(enabled=getattr(self.config, 'use_amp', True)):
                # If DDP is used, self.model is DDP-wrapped; access original model via .module
                model_to_call = self.model.module if dist.is_initialized() else self.model
                
                # Forward pass through the vision encoder and then the full RAG model
                features, cls_logits = model_to_call.vision_encoder(images, return_both=True)
                outputs = model_to_call(images, target_reports=reports, precomputed_features=features)
                
                # Calculate losses: generation loss and classification loss
                gen_loss = outputs["loss"] # From language model part
                cls_loss = F.binary_cross_entropy_with_logits(cls_logits, labels) # From vision encoder's classifier head
                
                # Combine losses (example: weighted sum)
                loss = gen_loss + getattr(self.config, 'cls_loss_weight', 0.3) * cls_loss
            
            # Scales loss. Calls backward() on scaled loss to create scaled gradients.
            self.scaler.scale(loss).backward()
            
            # Optional: Gradient clipping (unscale first, then clip, then scaler.step)
            if hasattr(self.config, 'clip_grad_norm') and self.config.clip_grad_norm > 0:
                self.scaler.unscale_(self.optimizer) # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called.
            # Otherwise, optimizer.step() is skipped.
            self.scaler.step(self.optimizer)
            # Updates the scale for next iteration.
            self.scaler.update()
            
            # Update dynamic sparse layers if they are part of the model
            # This loop might be slow if model is very deep; optimize if it becomes a bottleneck.
            for module in model_to_call.modules():
                if isinstance(module, DynamicSparseLayer):
                    module.update_mask() # This method should handle its own logic
            
            return loss.item(), outputs.get("generated_reports", []) # Return loss and generated reports
        
        except RuntimeError as e: # Handle runtime errors, especially OOM
            logger.error(f"Runtime error during training step: {e}", exc_info=True)
            if "out of memory" in str(e).lower():
                logger.critical("CUDA out of memory during training step. Training cannot proceed with current settings.", exc_info=True)
                # Attempt to clear cache, though it might not be enough for OOM during a step.
                torch.cuda.empty_cache() 
            raise # Re-raise to be caught by the main training loop for potential recovery or termination
        except Exception as e:
            logger.error(f"Unexpected error during training step: {e}", exc_info=True)
            raise

    def validate(self) -> Tuple[float, List[str], List[str]]:
        """
        Performs validation on the validation dataset.

        Iterates through the validation loader, computes loss and generates reports.
        Handles metric aggregation if in a distributed setup.

        Returns:
            Tuple[float, List[str], List[str]]: A tuple containing:
                - Average validation loss.
                - List of all generated reports from the validation set.
                - List of all reference reports from the validation set.
        """
        logger.info("Starting validation phase...")
        self.model.eval() # Set model to evaluation mode
        total_loss = 0.0
        all_generated_reports = [] # Store all generated reports for evaluation
        all_reference_reports = [] # Store all reference reports
        num_samples_processed = 0
        
        # Access underlying model if DDP-wrapped
        model_to_call = self.model.module if dist.is_initialized() else self.model

        with torch.no_grad(): # Disable gradient calculations for validation
            for batch_idx, batch in enumerate(self.val_loader):
                try:
                    images = batch['images'].to(self.device, non_blocking=True)
                    labels = batch['labels'].to(self.device, non_blocking=True) # May not be used for loss if model doesn't output cls_loss in eval
                    reports = batch['report'] # Ground truth reports

                    # AMP context for consistency, though often not strictly needed for eval if no backward pass
                    with autocast(enabled=getattr(self.config, 'use_amp', True)):
                        # Model forward pass for validation/inference
                        # Ensure the model's forward pass when target_reports is provided behaves correctly in eval mode
                        # (e.g., it should still calculate loss if possible, and generate reports)
                        outputs = model_to_call(images, target_reports=reports) 
                        
                        # Accumulate loss if provided by the model output
                        loss = outputs.get("loss") 
                        if loss is not None:
                            total_loss += loss.item() * images.size(0) # Weighted by batch size
                        elif batch_idx == 0: # Log once if loss is missing
                            logger.warning("Validation forward pass did not return 'loss'. Average validation loss will be inaccurate (0 or NaN).")
                        
                        # Collect generated and reference reports
                        generated_batch_reports = outputs.get("generated_reports", [])
                        all_generated_reports.extend(generated_batch_reports)
                        all_reference_reports.extend(reports)
                        num_samples_processed += images.size(0)

                except RuntimeError as e:
                    logger.error(f"Runtime error during validation batch {batch_idx}: {e}", exc_info=True)
                    if "out of memory" in str(e).lower(): # OOM during validation
                        logger.error("CUDA out of memory during validation. Consider reducing validation batch size.", exc_info=True)
                        # Decide if this is critical enough to stop validation or just skip batch.
                    # Continue to next batch to get partial results if possible.
                except Exception as e:
                    logger.error(f"Unexpected error during validation batch {batch_idx}: {e}", exc_info=True)
                    # Continue to next batch.
            
        # In distributed training, aggregate metrics from all processes
        if dist.is_initialized():
            logger.debug("Synchronizing validation metrics across DDP processes...")
            # Sum total_loss and num_samples_processed from all ranks
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            num_samples_tensor = torch.tensor(num_samples_processed, device=self.device)
            
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_samples_tensor, op=dist.ReduceOp.SUM)
            
            total_loss = total_loss_tensor.item()
            num_samples_processed = num_samples_tensor.item()

            # TODO: Implement proper gathering for lists (all_generated_reports, all_reference_reports)
            # `dist.all_gather_object` can be used for this, but requires careful handling.
            # For now, only the master process will have its local portion of these lists,
            # which means metrics like BLEU calculated on these will be partial in DDP.
            if not self.is_master():
                all_generated_reports = [] # Clear for non-master, or they'll have only their shard
                all_reference_reports = []
            else:
                logger.warning(
                    "In DDP mode, validation report lists (generated/references) are currently based on the master process's shard "
                    "or are partially gathered. For a full evaluation across all data in DDP, implement `dist.all_gather_object` "
                    "for these lists."
                )

        avg_loss = total_loss / num_samples_processed if num_samples_processed > 0 else float('inf')
        logger.info(f"Validation complete. Average Loss: {avg_loss:.4f} (total loss: {total_loss:.4f}, samples: {num_samples_processed}).")
        return avg_loss, all_generated_reports, all_reference_reports
    
    def train(self, epochs: int):
        """
        Main training loop.

        Iterates over the specified number of epochs, performing training steps
        and validation at the end of each epoch. Logs progress and metrics to
        wandb (if configured) and saves checkpoints.

        Args:
            epochs (int): The total number of epochs to train for.
        """
        logger.info(f"Starting training process for {epochs} epochs.")
        start_epoch = getattr(self.config, 'start_epoch', 0) # Allow resuming from a specific epoch
        
        for epoch in range(start_epoch, epochs):
            logger.info(f"Starting Epoch {epoch + 1}/{epochs}") # User-friendly 1-based epoch numbering
            
            # Set epoch for distributed sampler (shuffles data differently each epoch)
            if dist.is_initialized() and hasattr(self.train_loader.sampler, 'set_epoch'):
                logger.debug(f"Setting epoch {epoch} for training data sampler.")
                self.train_loader.sampler.set_epoch(epoch)
            
            epoch_train_loss = 0.0
            num_train_batches = 0
            
            # Training phase for the current epoch
            for batch_idx, batch in enumerate(self.train_loader):
                try:
                    loss, _ = self.train_step(batch) # Generated reports from train_step are not used here directly
                    epoch_train_loss += loss
                    num_train_batches += 1
                    
                    # Log batch-level training loss and learning rate
                    log_interval = getattr(self.config, 'log_interval', 10)
                    if self.is_master() and (batch_idx + 1) % log_interval == 0:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        logger.info(
                            f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{len(self.train_loader)} | "
                            f"Train Batch Loss: {loss:.4f} | LR: {current_lr:.2e}"
                        )
                        if wandb.run: # Check if wandb integration is active
                            wandb.log({
                                "train/batch_loss": loss,
                                "train/learning_rate": current_lr,
                                "trainer/global_step": epoch * len(self.train_loader) + batch_idx 
                            }, step=epoch * len(self.train_loader) + batch_idx) # Use a global step counter for wandb
                
                except RuntimeError as e: # Handle errors from train_step
                    logger.error(f"Runtime error during training epoch {epoch + 1}, batch {batch_idx + 1}: {e}", exc_info=True)
                    if "out of memory" in str(e).lower():
                        logger.critical("CUDA OOM during training. Training cannot continue with current settings.", exc_info=True)
                        if self.is_master(): # Save a rescue checkpoint from master
                            self.save_checkpoint(epoch, float('inf'), is_rescue=True)
                        raise # Re-raise OOM to stop the training process
                    logger.warning(f"Skipping batch {batch_idx + 1} in epoch {epoch + 1} due to runtime error.")
                    continue # Skip to the next batch
                except Exception as e:
                    logger.error(f"Unexpected error during training epoch {epoch + 1}, batch {batch_idx + 1}: {e}", exc_info=True)
                    logger.warning(f"Skipping batch {batch_idx + 1} in epoch {epoch + 1} due to unexpected error.")
                    continue
            
            avg_epoch_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else float('inf')
            
            # Validation phase at the end of the epoch
            logger.info(f"Starting validation for Epoch {epoch + 1}.")
            val_loss, val_generated_reports, val_reference_reports = self.validate() # Reports might be used for qualitative eval later
            
            # Log epoch-level metrics (master process only)
            if self.is_master():
                logger.info(f"Epoch {epoch + 1}/{epochs} Summary: Avg Train Loss: {avg_epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                if wandb.run:
                    wandb.log({
                        "epoch/train_loss": avg_epoch_train_loss,
                        "epoch/val_loss": val_loss,
                        "epoch": epoch + 1 # Log 1-based epoch
                    }, step= (epoch + 1) * len(self.train_loader) ) # Log at end of epoch's global steps
                
                # Save checkpoint based on configured interval or if it's the best model so far
                save_interval = getattr(self.config, 'save_checkpoint_interval', 1)
                if (epoch + 1) % save_interval == 0:
                    self.save_checkpoint(epoch + 1, val_loss) 
            
            # TODO: Implement early stopping logic if desired, based on validation loss.
            
        logger.info("Training process completed.")
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_rescue: bool = False):
        """
        Saves a model checkpoint.

        Only the master process saves the checkpoint. The checkpoint includes model state,
        optimizer state, scaler state (for AMP), epoch, validation loss, and the configuration.

        Args:
            epoch (int): The current epoch number (1-based for user-friendliness in filename).
            val_loss (float): The validation loss at this epoch.
            is_rescue (bool, optional): If True, saves as a "rescue" checkpoint,
                                        often used in OOM situations. Defaults to False.
        """
        if not self.is_master(): # Ensure only the master process saves checkpoints
            return
        
        try:
            # Get model state dictionary (handles DDP model automatically)
            model_state_dict = self.model.module.state_dict() if dist.is_initialized() else self.model.state_dict()
            
            # Create checkpoint dictionary
            checkpoint = {
                'epoch': epoch, # Current epoch (e.g., after completion)
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(), # Important for resuming AMP training
                'val_loss': val_loss, # Validation loss at this checkpoint
                'config': self.config # Save the training configuration
            }
            
            # Determine filename
            if is_rescue:
                filename = f"checkpoint_rescue_epoch_{epoch}.pt"
                logger.warning(f"Saving rescue checkpoint: {filename}")
            else:
                filename = f"checkpoint_epoch_{epoch}_valloss_{val_loss:.4f}.pt" # Include val_loss in name
            
            checkpoint_path = self.config.checkpoint_dir / filename
            # Ensure the checkpoint directory exists (should be handled by Config, but good practice)
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save the checkpoint
            torch.save(checkpoint, str(checkpoint_path))
            logger.info(f"Checkpoint saved successfully: {checkpoint_path}")

            # Optional: Save a 'latest_checkpoint.pt' for easy loading of the most recent model
            if not is_rescue: # Avoid overwriting latest with a rescue checkpoint unless intended
                latest_path = self.config.checkpoint_dir / "latest_checkpoint.pt"
                torch.save(checkpoint, str(latest_path))
                logger.debug(f"Checkpoint also updated/saved as: {latest_path}")

        except IOError as e:
            logger.error(f"IOError while saving checkpoint for epoch {epoch}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to save checkpoint for epoch {epoch}: {e}", exc_info=True)
    
    def cleanup(self):
        """
        Cleans up resources, particularly for distributed training and wandb.
        """
        logger.info("Starting cleanup process for DistributedRadiologyTrainer...")
        
        # Clean up distributed training environment
        if dist.is_initialized():
            logger.info("Cleaning up distributed training environment (A100Optimizer.cleanup_distributed will be called).")
            A100Optimizer.cleanup_distributed() # This method contains its own logging
        
        # Finish Weights & Biases run if active and on master process
        if self.is_master():
            if wandb.run: # Check if wandb.run is an active run object
                logger.info("Finishing Weights & Biases run...")
                try:
                    wandb.finish()
                    logger.info("Wandb run finished successfully.")
                except Exception as e:
                    logger.error(f"Error during wandb.finish(): {e}", exc_info=True)
            else:
                logger.info("Wandb was not active or already finished, no wandb.finish() call needed.")
        logger.info("Cleanup process for DistributedRadiologyTrainer completed.")