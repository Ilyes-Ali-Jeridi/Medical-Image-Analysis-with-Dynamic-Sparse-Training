import os
import torch
import torch.distributed as dist
import logging
import gc

logger = logging.getLogger(__name__)

class A100Optimizer:
    """Optimizations specific to the A100 GPU and distributed training."""
    @staticmethod
    def configure_device():
        """Configure device settings optimized for A100 GPUs."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
            
        # Enable TF32 and optimize memory usage for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set device and memory settings
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        
        # Correct way to set memory fraction for A100
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        # Enable CUDA graphs for repeated operations
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
        
        # Empty cache and trigger garbage collection
        torch.cuda.empty_cache()
        gc.collect()
        
        return torch.device(f"cuda:{local_rank}")
    
    @staticmethod
    def configure_distributed():
        """Configure distributed training settings."""
        try:
            if dist.is_initialized():
                return
                
            if "LOCAL_RANK" not in os.environ:
                os.environ["LOCAL_RANK"] = "0"
                
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            
            # Use NCCL backend for optimal GPU communication
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=local_rank
            )
            
            # Set NCCL parameters for A100
            os.environ["NCCL_DEBUG"] = "INFO"
            os.environ["NCCL_IB_DISABLE"] = "0"
            os.environ["NCCL_IB_GID_INDEX"] = "3"
            os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
            
            logger.info(f"Distributed training initialized: rank {local_rank}/{world_size-1}")
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {str(e)}")
            raise
    
    @staticmethod
    def cleanup_distributed():
        """Clean up distributed training resources."""
        if dist.is_initialized():
            dist.destroy_process_group()
            
    @staticmethod
    def optimize_performance():
        """Apply A100-specific performance optimizations."""
        # Enable TF32 for faster matrix multiplications on A100
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN autotuner
        torch.backends.cudnn.benchmark = True
        
        # Disable deterministic algorithms for better performance
        torch.backends.cudnn.deterministic = False
        
        # Set optimal thread settings for NC48ads
        torch.set_num_threads(48)
        
        # Enable CUDA graphs for repeated operations
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
            torch.cuda.synchronize()