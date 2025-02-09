from pathlib import Path
import torch

class Config:
    """Configuration for the medical imaging system."""
    def __init__(self):
        # GPU and distributed training settings
        self.num_gpus = torch.cuda.device_count()
        self.batch_size = 128 * self.num_gpus  # Scale with number of GPUs
        self.gradient_accumulation_steps = 4
        self.num_workers = 8 * self.num_gpus
        self.pin_memory = True
        
        # Model architecture details
        self.image_size = 256
        self.patch_size = 16
        self.embed_dim = 768
        self.num_layers = 12
        self.num_heads = 8
        self.sparse_rate = 0.4
        
        # Training parameters
        self.base_learning_rate = 2e-4
        self.learning_rate = self.base_learning_rate * self.num_gpus  # Scale with GPUs
        self.weight_decay = 0.01
        self.max_epochs = 10
        self.warmup_steps = 1000
        self.gradient_clip = 1.0
        
        # Paths
        self.data_dir = Path("/data/MIMICCXR")
        self.output_dir = Path("outputs")
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        
        # Create directories
        for directory in [self.data_dir, self.output_dir, self.checkpoint_dir, self.log_dir]:
            directory.mkdir(parents=True, exist_ok=True)