from pathlib import Path
import torch

class Config:
    def __init__(self):
        # Hardware Settings
        self.num_gpus = torch.cuda.device_count()
        self.batch_size = 64  
        self.gradient_accumulation_steps = 2
        self.num_workers = min(8, os.cpu_count() // 2)
        
        # Model Architecture
        self.image_size = 224  # Reduced for better memory usage
        self.patch_size = 16
        self.embed_dim = 768
        self.num_layers = 12
        self.sparse_rate = 0.4
        
        # Dynamic Learning Rate
        self.base_lr = 2e-4
        self.lr = self.base_lr * self.batch_size * self.gradient_accumulation_steps / 256
        self.weight_decay = 0.01
        
        # Paths (Safe Directory Creation)
        self.data_dir = Path("/data/MIMICCXR")
        self.output_dir = Path("outputs")
        self.checkpoint_dir = self.output_dir / "checkpoints"
        
        # Create directories safely
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
