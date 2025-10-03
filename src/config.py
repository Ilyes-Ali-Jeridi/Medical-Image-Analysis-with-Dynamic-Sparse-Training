from pathlib import Path
import torch
import os # For os.cpu_count()

class Config:
    """
    Configuration class for the medical imaging training and deployment pipeline.

    This class centralizes all hyperparameters, path settings, and other
    configurable parameters for the application.
    """
    def __init__(self):
        # Hardware Settings
        self.num_gpus = torch.cuda.device_count()  # Number of available GPUs
        self.batch_size = 64  # Batch size per GPU
        self.gradient_accumulation_steps = 2  # Accumulate gradients over N steps to simulate larger batch size
        # Number of data loader worker processes. Uses half of CPU cores, capped at 8.
        # Fallbacks to 4 if cpu_count is not available.
        cpu_cores = os.cpu_count()
        self.num_workers = min(8, cpu_cores // 2) if cpu_cores else 4
        self.cuda_memory_fraction = 0.95 # Per-process GPU memory fraction to set (e.g., for A100s)
        # PyTorch number of threads. Default to os.cpu_count() for general portability,
        # fallback to a reasonable number if cpu_count() is not available or returns a very small number.
        self.torch_num_threads = cpu_cores if cpu_cores and cpu_cores > 1 else 48 # Fallback to 48 if cpu_count is problematic
        
        # Model Architecture
        # These parameters define the architecture of the vision and language models.
        self.image_size = 224  # Input image size (square images assumed)
        self.patch_size = 16   # Patch size for Vision Transformer (ViT)
        self.embed_dim = 768   # Embedding dimension for ViT and language model
        self.num_layers = 12   # Number of transformer layers in ViT
        self.sparse_rate = 0.4 # Sparsity rate for dynamic sparse layers
        
        # Dynamic Learning Rate Configuration
        # The learning rate is scaled based on batch size and gradient accumulation.
        self.base_lr = 2e-4  # Base learning rate
        # Effective learning rate, scaled by total effective batch size relative to a base of 256
        self.lr = self.base_lr * self.batch_size * self.gradient_accumulation_steps / 256
        self.weight_decay = 0.01  # Weight decay for optimizer
        
        # Paths Configuration
        # Defines all relevant input/output directories and files.
        # These paths are created if they don't exist during initialization.
        self.data_dir = Path("/data/MIMICCXR")  # Root directory for input datasets
        self.output_dir = Path("outputs")      # Root directory for all generated outputs
        
        self.checkpoint_dir = self.output_dir / "checkpoints"  # Directory to save model checkpoints
        self.checkpoint_save_every_n_epochs = 5 # Frequency for saving periodic checkpoints (e.g., every N epochs)
        self.save_best_checkpoint = True # Flag to enable/disable saving of the best model checkpoint based on validation loss
        self.cache_dir = Path("/tmp/mimic_cache")             # Directory for caching processed data samples
        self.log_file = self.output_dir / "training.log"     # Path to the main log file

        # Create directories safely if they don't already exist
        # This ensures that the application can write to these locations.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True) # Ensure log file's parent directory exists
