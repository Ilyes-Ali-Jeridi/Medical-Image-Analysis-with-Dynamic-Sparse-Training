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
    """Distributed trainer for medical imaging with proper GPU utilization."""
    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        config: Any,
        device: torch.device = None
    ):
        self.config = config
        self.device = device or A100Optimizer.configure_device()
        self.setup_distributed()
        
        # Initialize model and move to GPU
        vision_encoder = MedicalViT(sparse_rate=config.sparse_rate)
        self.model = MedicalRAG(vision_encoder=vision_encoder, sparse_rate=config.sparse_rate)
        self.model.to(self.device)
        
        if dist.is_initialized():
            self.model = DDP(
                self.model,
                device_ids=[dist.get_rank()],
                output_device=dist.get_rank(),
                find_unused_parameters=False
            )
        
        # Initialize optimizer with correct learning rate scaling
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scaler = GradScaler()
        self.setup_data_loaders(train_csv, val_csv)
        
        # Initialize wandb for the master process only
        if self.is_master():
            wandb.init(project="medical-training", config=vars(config))
    
    def setup_distributed(self):
        """Sets up distributed training environment."""
        if not dist.is_initialized():
            A100Optimizer.configure_distributed()
    
    def setup_data_loaders(self, train_csv: str, val_csv: str):
        """Initializes data loaders with proper distributed sampling."""
        train_dataset = MIMICCXRDataset(train_csv)
        val_dataset = MIMICCXRDataset(val_csv)
        
        train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if dist.is_initialized() else None
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            sampler=val_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
    
    @staticmethod
    def is_master() -> bool:
        """Checks if this is the master process."""
        return not dist.is_initialized() or dist.get_rank() == 0
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, List[str]]:
        """Performs a single training step with gradient scaling."""
        self.model.train()
        self.optimizer.zero_grad()
        
        images = batch['images'].to(self.device)
        
        with autocast():
            # Get both features and classification logits
            features, cls_logits = self.model.module.vision_encoder(images, return_both=True)
            outputs = self.model.module(images, target_reports=batch['report'], precomputed_features=features)
            
            gen_loss = outputs["loss"]
            cls_loss = F.binary_cross_entropy_with_logits(
                cls_logits,
                batch['labels'].to(self.device)
            )
            loss = gen_loss + 0.3 * cls_loss
        
        # Scale loss and backpropagate
        self.scaler.scale(loss).backward()
        
        # Update dynamic sparse layers
        if dist.is_initialized():
            for module in self.model.module.modules():
                if isinstance(module, DynamicSparseLayer):
                    module.update_mask()
        else:
            for module in self.model.modules():
                if isinstance(module, DynamicSparseLayer):
                    module.update_mask()
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item(), outputs["generated_reports"]
    
    def validate(self) -> Tuple[float, List[str], List[str]]:
        """Performs validation with proper error handling."""
        self.model.eval()
        total_loss = 0.0
        all_generated = []
        all_references = []
        count = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    images = batch['images'].to(self.device)
                    outputs = self.model.module(images, target_reports=batch['report'])
                    loss = outputs["loss"].item()
                    total_loss += loss * images.size(0)
                    count += images.size(0)
                    all_generated.extend(outputs["generated_reports"])
                    all_references.extend(batch['report'])
                except Exception as e:
                    logger.error(f"Validation error: {str(e)}")
                    continue
        
        # Gather results across all processes
        if dist.is_initialized():
            total_loss = torch.tensor(total_loss).to(self.device)
            count = torch.tensor(count).to(self.device)
            dist.all_reduce(total_loss)
            dist.all_reduce(count)
            total_loss = total_loss.item()
            count = count.item()
        
        avg_loss = total_loss / count if count > 0 else float('inf')
        return avg_loss, all_generated, all_references
    
    def train(self, epochs: int = 10):
        """Main training loop with proper logging and checkpointing."""
        for epoch in range(epochs):
            if dist.is_initialized():
                self.train_loader.sampler.set_epoch(epoch)
            
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                try:
                    loss, gen_reports = self.train_step(batch)
                    epoch_loss += loss
                    num_batches += 1
                    
                    if self.is_master() and batch_idx % 10 == 0:
                        logger.info(f"Epoch {epoch} Batch {batch_idx} Loss: {loss:.4f}")
                        wandb.log({
                            "batch_loss": loss,
                            "epoch": epoch,
                            "batch": batch_idx
                        })
                except Exception as e:
                    logger.error(f"Training error in epoch {epoch}, batch {batch_idx}: {str(e)}")
                    continue
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
            
            # Validation
            val_loss, gen_reports, ref_reports = self.validate()
            
            if self.is_master():
                logger.info(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
                logger.info(f"Epoch {epoch} Validation Loss: {val_loss:.4f}")
                
                # Log to wandb
                wandb.log({
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_loss": val_loss
                })
                
                # Save checkpoint
                if (epoch + 1) % 5 == 0:
                    self.save_checkpoint(epoch, val_loss)
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        """Saves model checkpoint with proper error handling."""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict() if dist.is_initialized() else self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
                'config': self.config
            }
            
            checkpoint_path = f"{self.config.checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
    
    def cleanup(self):
        """Cleanup distributed training resources."""
        if dist.is_initialized():
            A100Optimizer.cleanup_distributed()
        if self.is_master():
            wandb.finish()