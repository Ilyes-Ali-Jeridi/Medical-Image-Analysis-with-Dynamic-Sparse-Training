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
    Handles distributed training for the medical imaging system.
    This class orchestrates the model, data loaders, optimizer, and training loop,
    ensuring seamless operation across multiple GPUs.
    """
    def __init__(self, train_csv: str, val_csv: str, config: Any, device: torch.device = None):
        self.config = config
        self.device = device or A100Optimizer.configure_device()
        self.setup_distributed()
        
        # Initialize model and move to GPU
        vision_encoder = MedicalViT(config)
        self.model = MedicalRAG(config, vision_encoder=vision_encoder)
        self.model.to(self.device)
        
        if dist.is_initialized():
            self.model = DDP(
                self.model,
                device_ids=[dist.get_rank()],
                output_device=dist.get_rank(),
                find_unused_parameters=False # Optimized for performance
            )
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        self.scaler = GradScaler()
        self.setup_data_loaders(train_csv, val_csv)
        
        if self.is_master():
            wandb.init(project="medical-imaging-rag", config=vars(config))
    
    @property
    def model_without_ddp(self):
        """Provides access to the underlying model instance, bypassing DDP."""
        return self.model.module if isinstance(self.model, DDP) else self.model

    def setup_distributed(self):
        """Initializes the distributed training environment."""
        if not dist.is_initialized():
            A100Optimizer.configure_distributed()
    
    def setup_data_loaders(self, train_csv: str, val_csv: str):
        """Initializes data loaders with distributed samplers."""
        self.train_dataset = MIMICCXRDataset(train_csv)
        self.val_dataset = MIMICCXRDataset(val_csv)
        
        train_sampler = DistributedSampler(self.train_dataset) if dist.is_initialized() else None
        val_sampler = DistributedSampler(self.val_dataset, shuffle=False) if dist.is_initialized() else None
        
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, sampler=train_sampler,
            num_workers=self.config.num_workers, pin_memory=True, drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.config.batch_size, sampler=val_sampler,
            num_workers=self.config.num_workers, pin_memory=True
        )
    
    def is_master(self) -> bool:
        """Checks if the current process is the master process."""
        return not dist.is_initialized() or dist.get_rank() == 0
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, List[str]]:
        """Performs a single training step, including forward pass, loss computation, and backpropagation."""
        self.model.train()
        self.optimizer.zero_grad()
        
        images = batch['images'].to(self.device)
        
        with autocast():
            features, cls_logits = self.model_without_ddp.vision_encoder(images, return_both=True)
            outputs = self.model_without_ddp(images, target_reports=batch['report'], precomputed_features=features)
            
            gen_loss = outputs["loss"]
            cls_loss = F.binary_cross_entropy_with_logits(cls_logits, batch['labels'].to(self.device))
            loss = gen_loss + self.config.classification_loss_weight * cls_loss
        
        self.scaler.scale(loss).backward()
        
        for module in self.model_without_ddp.modules():
            if isinstance(module, DynamicSparseLayer):
                module.update_mask()
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item(), outputs["generated_reports"]
    
    def validate(self) -> Tuple[float, List[str], List[str], List[torch.Tensor]]:
        """Performs validation on the validation set."""
        self.model.eval()
        total_loss, count = 0.0, 0
        all_generated, all_references, all_images = [], [], []
        
        with torch.no_grad():
            for batch in self.val_loader:
                try:
                    images = batch['images'].to(self.device)
                    outputs = self.model_without_ddp(images, target_reports=batch['report'])

                    total_loss += outputs["loss"].item() * images.size(0)
                    count += images.size(0)

                    all_generated.extend(outputs["generated_reports"])
                    all_references.extend(batch['report'])
                    all_images.append(images.cpu())
                except Exception as e:
                    logger.error(f"Validation error: {str(e)}")
        
        if dist.is_initialized():
            # This part needs a more sophisticated implementation for gathering lists across processes
            pass
        
        avg_loss = total_loss / count if count > 0 else float('inf')
        return avg_loss, all_generated, all_references, all_images
    
    def train(self, epochs: int):
        """The main training loop, iterating over epochs and batches."""
        for epoch in range(epochs):
            if dist.is_initialized():
                self.train_loader.sampler.set_epoch(epoch)
            
            epoch_loss, num_batches = 0.0, 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                try:
                    loss, _ = self.train_step(batch)
                    epoch_loss += loss
                    num_batches += 1
                    
                    if self.is_master() and batch_idx % 10 == 0:
                        logger.info(f"Epoch {epoch}/{epochs-1} | Batch {batch_idx}/{len(self.train_loader)} | Loss: {loss:.4f}")
                        wandb.log({"batch_loss": loss, "epoch": epoch, "batch": batch_idx})
                except Exception as e:
                    logger.error(f"Training error at epoch {epoch}, batch {batch_idx}: {str(e)}")
            
            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
            val_loss, _, _, _ = self.validate()
            
            if self.is_master():
                logger.info(f"Epoch {epoch} Summary: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": val_loss})
                
                if (epoch + 1) % 5 == 0:
                    self.save_checkpoint(epoch, val_loss)
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        """Saves a model checkpoint."""
        try:
            checkpoint_path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model_without_ddp.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
                'config': self.config
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
    
    def get_validation_data(self) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        """Returns validation data for evaluation."""
        _, generated, references, images = self.validate()
        return generated, references, images

    def cleanup(self):
        """Cleans up resources from distributed training."""
        if dist.is_initialized():
            A100Optimizer.cleanup_distributed()
        if self.is_master():
            wandb.finish()