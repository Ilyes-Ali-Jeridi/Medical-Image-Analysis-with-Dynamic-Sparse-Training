import os
import torch
import torch.distributed as dist
import logging
import gc

logger = logging.getLogger(__name__)

class A100Optimizer:
    """
    A utility class providing static methods to configure and optimize PyTorch
    for the NVIDIA A100 GPU, particularly in a distributed training context.
    """
    @staticmethod
    def configure_device() -> torch.device:
        """
        Configures the device and CUDA settings for optimal A100 performance.

        This includes enabling TF32, setting memory fractions, and cleaning up memory.

        Returns:
            torch.device: The configured CUDA device.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot configure A100.")
            
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        
        # Set memory fraction to avoid fragmentation and allow for overhead
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"Device configured for A100 at local rank {local_rank}.")
        return torch.device(f"cuda:{local_rank}")
    
    @staticmethod
    def configure_distributed():
        """
        Initializes the distributed process group using the NCCL backend.

        Sets environment variables for optimal NCCL performance on A100 GPUs
        with InfiniBand interconnects.
        """
        try:
            if dist.is_initialized():
                return
                
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=local_rank
            )
            
            # NCCL environment variables for performance tuning
            os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
            os.environ["NCCL_DEBUG"] = "INFO"
            
            logger.info(f"Distributed training initialized: rank {local_rank}/{world_size-1}")
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            raise
    
    @staticmethod
    def cleanup_distributed():
        """
        Cleans up and destroys the distributed process group.
        """
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed.")
            
    @staticmethod
    def optimize_performance():
        """
        Applies global PyTorch settings for maximizing performance on A100 GPUs.

        This should be called once at the beginning of the main script.
        """
        # Enable TF32, which provides a significant performance boost on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable the cuDNN autotuner to find the best algorithm for the hardware
        torch.backends.cudnn.benchmark = True
        
        # Disabling deterministic algorithms can improve performance
        torch.backends.cudnn.deterministic = False
        
        logger.info("Global performance settings for A100 have been applied.")