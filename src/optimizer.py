import os
import torch
import torch.distributed as dist
import logging
import gc # For garbage collection

logger = logging.getLogger(__name__)

class A100Optimizer:
    """
    Provides static methods for configuring PyTorch and distributed training settings,
    with a focus on optimizations for NVIDIA A100 GPUs.

    These methods aim to enhance performance by leveraging A100-specific features
    like TF32, and by setting up distributed training environments correctly.
    While named for A100, many settings are beneficial for modern NVIDIA GPUs.
    """
    @staticmethod
    def configure_device(cuda_memory_fraction: float = 0.95) -> torch.device:
        """
        Configures PyTorch settings for optimal performance on the current CUDA device.

        This includes enabling TensorFloat-32 (TF32) for faster matrix math,
        setting cuDNN benchmark and deterministic flags, configuring the CUDA device
        based on LOCAL_RANK, attempting to set per-process memory fraction,
        and clearing CUDA cache.

        Args:
            cuda_memory_fraction (float): The fraction of GPU memory to allocate per process.
                                          Default is 0.95. Set to 1.0 to disable.

        Returns:
            torch.device: The configured PyTorch device (e.g., 'cuda:0').

        Raises:
            RuntimeError: If CUDA is not available.
            ValueError: If LOCAL_RANK environment variable is invalid.
        """
        logger.info("Configuring device settings for GPU...")
        if not torch.cuda.is_available():
            logger.error("CUDA is not available. This optimizer requires a CUDA-enabled GPU.")
            raise RuntimeError("CUDA is not available. GPU required.")
            
        # Enable TF32 for matrix multiplications and cuDNN operations on Ampere and newer GPUs.
        # TF32 provides a performance boost with minimal precision loss for many deep learning workloads.
        logger.debug("Enabling TF32 for torch.backends.cuda.matmul and torch.backends.cudnn.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # cuDNN benchmark mode enables cuDNN to search for the optimal algorithm for the specific input sizes.
        # This can improve performance but might make results slightly non-deterministic if input sizes vary.
        logger.debug("Enabling cuDNN benchmark mode (torch.backends.cudnn.benchmark = True).")
        torch.backends.cudnn.benchmark = True
        # Setting deterministic to False can improve performance by allowing non-deterministic algorithms.
        # For full reproducibility, this might need to be True, but often False is preferred for speed.
        torch.backends.cudnn.deterministic = False 
        logger.debug("Set cuDNN deterministic mode to False.")
        
        # Determine local rank from environment variable, typically set by distributed launchers.
        local_rank_str = os.environ.get("LOCAL_RANK")
        if local_rank_str is None:
            logger.warning("LOCAL_RANK environment variable not set. Defaulting to device 0 for non-distributed context.")
            local_rank = 0
        else:
            try:
                local_rank = int(local_rank_str)
            except ValueError:
                logger.error(f"Invalid LOCAL_RANK: '{local_rank_str}'. Must be an integer. Defaulting to 0.")
                local_rank = 0 # Fallback, though this situation might indicate a setup error for DDP.
        
        logger.info(f"Setting CUDA device to local_rank: {local_rank}.")
        torch.cuda.set_device(local_rank) # Set the current CUDA device
        
        # Attempt to set per-process memory fraction.
        # Useful on shared systems or multi-process single-GPU setups to manage memory.
        # A value of 1.0 typically means no specific fraction is set by PyTorch this way.
        if 0.0 < cuda_memory_fraction < 1.0: # Only set if fraction is not trivial
            try:
                torch.cuda.set_per_process_memory_fraction(cuda_memory_fraction, local_rank) 
                logger.debug(f"Successfully set per-process memory fraction to {cuda_memory_fraction} on device {local_rank}.")
            except Exception as mem_e: # Catch broad errors as this can fail on some systems/drivers
                logger.warning(f"Could not set per-process memory fraction to {cuda_memory_fraction} on device {local_rank}: {mem_e}. "
                               "This might be fine on some hardware or if driver doesn't support it.", exc_info=True)
        else:
            logger.info(f"Per-process memory fraction not set (value: {cuda_memory_fraction}). PyTorch will use its default memory management.")

        # Check for CUDA Graph capability (Compute Capability 7.0+).
        # CUDA Graphs can significantly speed up repeated operations by reducing launch overhead.
        # Note: Actual usage of CUDA Graphs must be implemented explicitly in the model's training/inference loop.
        if torch.cuda.get_device_capability(local_rank)[0] >= 7:
            logger.debug(f"Device {local_rank} supports CUDA Graphs (Compute Capability >= 7.0). "
                         "Consider explicit CUDA Graph implementation for performance critical sections.")
        
        # Clear CUDA cache and run Python garbage collection to free up memory.
        logger.debug("Emptying CUDA cache and triggering Python garbage collection.")
        torch.cuda.empty_cache()
        gc.collect()
        
        device = torch.device(f"cuda:{local_rank}")
        logger.info(f"Device configuration complete. Active device: {device}")
        return device
    
    @staticmethod
    def configure_distributed():
        """
        Configures the distributed training environment using `torch.distributed`.

        Initializes the process group for distributed training, typically using the NCCL backend
        for GPU communication. Relies on environment variables like LOCAL_RANK, WORLD_SIZE,
        MASTER_ADDR, and MASTER_PORT, which should be set by the distributed launcher
        (e.g., torchrun, Slurm).
        NCCL-specific environment variables (like NCCL_DEBUG, NCCL_IB_DISABLE) are
        expected to be set externally via the launch environment if needed, for greater flexibility.

        Raises:
            ValueError: If LOCAL_RANK or WORLD_SIZE environment variables are invalid or missing when expected.
            RuntimeError: If `dist.init_process_group` fails.
            Exception: For other unexpected errors during setup.
        """
        logger.info("Configuring distributed training environment...")
        try:
            if dist.is_initialized():
                logger.info("Distributed training environment is already initialized.")
                return
                
            # LOCAL_RANK is crucial for identifying the current process's GPU.
            local_rank_env = os.environ.get("LOCAL_RANK")
            if local_rank_env is None:
                logger.warning("LOCAL_RANK environment variable not set. "
                               "Assuming single-process mode or manual distributed setup. "
                               "Skipping torch.distributed.init_process_group by this method.")
                return # Do not proceed if DDP is expected but LOCAL_RANK is missing.

            try:
                local_rank = int(local_rank_env)
            except ValueError:
                logger.error(f"Invalid LOCAL_RANK: '{local_rank_env}'. Distributed initialization failed.", exc_info=True)
                raise ValueError(f"Invalid LOCAL_RANK: {local_rank_env}")

            # WORLD_SIZE indicates the total number of processes participating.
            world_size_env = os.environ.get("WORLD_SIZE", "1") # Default to 1 for single-process case
            try:
                world_size = int(world_size_env)
            except ValueError:
                logger.error(f"Invalid WORLD_SIZE: '{world_size_env}'. Distributed initialization failed.", exc_info=True)
                raise ValueError(f"Invalid WORLD_SIZE: {world_size_env}")

            if world_size <= 1 and local_rank == 0:
                logger.info("World size is 1. Skipping NCCL backend initialization as it's not strictly needed for single process.")
                # If using DDP wrapper even for single GPU, it might still expect init.
                # However, if not using DDP, this is fine.
                # For this template, we'll assume if WORLD_SIZE=1, DDP is not the primary concern for this specific init.
                # If a user *wants* to init for DDP with world_size=1, they might need a different setup.
                # Alternatively, one could initialize with a dummy backend like 'gloo' if needed for DDP structure.
                return # Or initialize with a different backend if DDP structure is used for single GPU

            # Initialize the process group. `init_method='env://'` reads MASTER_ADDR, MASTER_PORT from env.
            logger.info(f"Initializing process group with backend 'nccl', init_method 'env://', world_size {world_size}, rank {local_rank}.")
            dist.init_process_group(
                backend="nccl",       # NCCL is generally recommended for NVIDIA GPU communication.
                init_method="env://", # Assumes MASTER_ADDR and MASTER_PORT are set in the environment.
                world_size=world_size,
                rank=local_rank
            )
            
            
            # NCCL environment variables (e.g., NCCL_DEBUG, NCCL_IB_DISABLE) are typically set
            # in the shell environment or launch script for more control, so they are not set here.
            # Users should configure these externally if specific NCCL behavior is needed.
            logger.info("NCCL environment variables should be configured externally if specific settings are required.")
            
            logger.info(f"Distributed training initialized successfully. Rank {dist.get_rank()} of {dist.get_world_size()}.")
        
        except RuntimeError as rt_e: # Catch specific errors from init_process_group
            logger.error(f"RuntimeError during distributed training initialization: {rt_e}. "
                         "Ensure MASTER_ADDR, MASTER_PORT, LOCAL_RANK, WORLD_SIZE are correctly set "
                         "and NCCL is properly installed and configured for your hardware.", exc_info=True)
            raise
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"An unexpected error occurred during distributed training initialization: {e}", exc_info=True)
            raise
    
    @staticmethod
    def cleanup_distributed():
        """
        Cleans up distributed training resources.

        If the process group was initialized, this method destroys it.
        """
        logger.info("Attempting to clean up distributed training resources...")
        if dist.is_initialized():
            current_rank = dist.get_rank()
            logger.info(f"Destroying distributed process group for rank {current_rank}.")
            dist.destroy_process_group()
            logger.info(f"Process group for rank {current_rank} destroyed successfully.")
        else:
            logger.info("Distributed training was not initialized or already cleaned up. No action taken.")
            
    @staticmethod
    def optimize_performance():
        """
        Applies general PyTorch performance optimizations, potentially beneficial for A100s.

        This includes enabling TF32, cuDNN benchmark mode, and setting PyTorch thread count
        based on the provided configuration. Some of these settings might overlap with
        `configure_device` but are reiterated here for emphasis or if used independently.

        Args:
            torch_num_threads (Optional[int]): Number of threads for PyTorch inter-op parallelism.
                                              If None, PyTorch's default is used.
                                              If 0, it might use a system-determined optimal number.
        """
        logger.info("Applying general PyTorch performance optimizations...")
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Skipping PyTorch performance optimizations that rely on CUDA.")
            return

        # Ensure TF32 is enabled for matmul and cuDNN operations on compatible hardware.
        logger.debug("Ensuring TF32 is enabled for matmul and cuDNN operations.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Ensure cuDNN benchmark mode is enabled for auto-tuning algorithms.
        logger.debug("Ensuring cuDNN benchmark mode is enabled.")
        torch.backends.cudnn.benchmark = True
        
        # Ensure deterministic mode is disabled for potential performance gains.
        logger.debug("Ensuring cuDNN deterministic mode is disabled for potential performance gain.")
        torch.backends.cudnn.deterministic = False
        
        # Set PyTorch number of threads for inter-operation parallelism.
        # A value of 0 typically lets PyTorch decide based on system resources.
        # Explicitly setting can sometimes be beneficial.
        if torch_num_threads is not None:
            if torch_num_threads > 0:
                logger.debug(f"Setting PyTorch number of threads to {torch_num_threads}.")
                torch.set_num_threads(torch_num_threads)
            else: # If 0 or negative, it might mean use default or system optimal
                logger.debug(f"torch_num_threads is {torch_num_threads}, relying on PyTorch default or system-determined number of threads.")
        else: # If None was passed
            logger.debug("torch_num_threads not specified, PyTorch will use its default thread settings.")
        
        # The torch.cuda.synchronize() call was removed as it's not a general performance optimization
        # and is more relevant for specific timing or debugging scenarios.
        # CUDA Graph capability logging is already handled in configure_device.
        logger.info("General PyTorch performance optimizations applied.")