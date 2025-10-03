"""
Main entry point for the medical image analysis and report generation pipeline.

This script handles:
1.  Parsing command-line arguments for configuration overrides.
2.  Setting up application-wide logging.
3.  Initializing the main `Config` object and applying overrides.
4.  Downloading necessary NLTK data.
5.  Coordinating the training, evaluation, and deployment phases based on arguments.
6.  Managing GPU/device configurations and distributed training setup via `A100Optimizer`.
7.  Cleaning up resources on exit.
"""
import os
import logging
import torch
from pathlib import Path
import argparse
import nltk

# Download NLTK data at startup
# Basic logging configuration for early messages, will be overridden by setup_logging later if successful.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Attempt to download 'punkt' and verify. Exit on failure as it's critical for some features.
try:
    logger_init = logging.getLogger(__name__) # Use a logger for this initial step too.
    logger_init.info("Checking/Downloading NLTK 'punkt' tokenizer data...")
    nltk.download('punkt', quiet=True) # Download 'punkt' if not already present.
    nltk.data.find('tokenizers/punkt') # Verify that 'punkt' can be found by NLTK.
    logger_init.info("NLTK 'punkt' data is available.")
except Exception as e: # Catch a broad range of exceptions during download or find.
    # Log a critical error and exit if NLTK 'punkt' data isn't available, as it's essential for BLEU scoring.
    logging.critical(
        f"Failed to download or verify NLTK 'punkt' data: {e}. "
        "This is critical for text processing functionalities (e.g., BLEU scoring in evaluation). "
        "Please check your internet connection, NLTK setup, and ensure 'punkt' can be downloaded/found. Exiting application.",
        exc_info=True # Include exception information in the log.
    )
    sys.exit(1) # Exit the script with an error code. Use sys.exit for clarity.

from config import Config
from trainer import DistributedRadiologyTrainer
from optimizer import A100Optimizer
from evaluator import MedicalEvaluator
from deployer import RadiologyDeployer

def setup_logging(log_file_path: Path):
    """
    Configures application-wide logging.

    Sets up a formatted logger that outputs to both a specified file and the console.
    It clears any existing handlers on the root logger to prevent duplicate logging
    if this function is called multiple times or if other libraries also configure logging.
    If the file handler cannot be created (e.g., due to permission issues),
    it falls back to basic console logging with a warning.

    Args:
        log_file_path (Path): The path to the log file where logs will be written.
    """
    # Get the root logger; all module loggers inherit from this.
    root_logger = logging.getLogger()
    
    # Clear any pre-existing handlers on the root logger to prevent duplicate messages
    # or interference from other libraries that might have configured logging.
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]: # Iterate over a copy
            root_logger.removeHandler(handler)
            handler.close() # Close handler to release resources like file locks
    
    # Set the desired logging level for the root logger.
    # This will affect all loggers unless they have their own level set.
    root_logger.setLevel(logging.INFO) # Default level, e.g., INFO, DEBUG
    
    # Define the format for log messages
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S' # Added date format
    )
    
    # File Handler: Writes log messages to a file.
    try:
        file_handler = logging.FileHandler(log_file_path, mode='a') # Append mode
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logging.info(f"Logging to file: {log_file_path}") # Log successful setup of file handler
    except IOError as e:
        # If file handler setup fails (e.g., permission denied), log a warning.
        # BasicConfig is used here as a fallback if root logger is now handler-less.
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.warning(
            f"Failed to set up log file handler at {log_file_path}: {e}. "
            "Logging to console only with potentially basic configuration.",
            exc_info=True
        )

    # Stream Handler: Writes log messages to the console (stderr by default).
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

def main():
    """
    Main function to run the medical image training, evaluation, and deployment pipeline.

    Parses command-line arguments, initializes configurations, sets up logging,
    and orchestrates the different operational phases (training, evaluation, deployment)
    based on the provided arguments.
    """
    # --- Argument Parsing ---
    # Defines command-line arguments to control various aspects of the pipeline.
    parser = argparse.ArgumentParser(
        description='Medical Image Analysis and Report Generation Pipeline. This script orchestrates the training, evaluation, and deployment of medical imaging models, allowing for flexible configuration via command-line arguments.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help message.
    )
    # Required arguments for specifying input data CSV files. These are filenames, expected to be found within the data_dir.
    parser.add_argument('--train-csv', type=str, required=True, 
                        help='Filename of the CSV metadata for training (e.g., "train_split.csv"). This file is expected to be located within the data directory specified by --data-dir or the default in Config.')
    parser.add_argument('--val-csv', type=str, required=True, 
                        help='Filename of the CSV metadata for validation (e.g., "val_split.csv"). This file is expected to be located within the data directory.')
    
    # Core operational parameters for training
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Total number of epochs for which the model will be trained.')
    
    # Optional flags to control which stages of the pipeline are executed
    parser.add_argument('--eval', action='store_true', 
                        help='If set, an evaluation run is performed on the validation set. This typically happens after training, or can be standalone if loading a pre-trained model.')
    parser.add_argument('--deploy', action='store_true', 
                        help='If set, the model is deployed (e.g., exported to ONNX format). This usually follows training and uses the latest/best checkpoint.')
    
    # Path override arguments: Allow users to specify custom locations for various directories and files.
    # Using type=Path for these arguments enables argparse to perform basic validation of path-like strings.
    parser.add_argument('--data-dir', type=Path, 
                        help='Override the default base directory for all input data. CSV files and image subdirectories are expected to be relative to this path.')
    parser.add_argument('--output-dir', type=Path, 
                        help='Override the default root directory for all generated outputs (e.g., "outputs/"). Checkpoints and log files will be placed in subdirectories here, unless their paths are also explicitly overridden.')
    parser.add_argument('--checkpoint-dir', type=Path, 
                        help='Override the default directory for saving model checkpoints during training and loading them for evaluation/deployment (e.g., "outputs/checkpoints/").')
    parser.add_argument('--cache-dir', type=Path, 
                        help='Override the default directory used for caching preprocessed dataset samples to speed up data loading (e.g., "/tmp/mimic_cache/").')
    parser.add_argument('--log-file', type=Path, 
                        help='Override the default path for the main log file (e.g., "outputs/training.log"). If a relative path is provided, it is typically interpreted as relative to the final output directory.')
    
    args = parser.parse_args()

    # --- Configuration Initialization and Overrides ---
    # Initialize the Config object, which holds default settings (including paths).
    config = Config() 

    # Override Config attributes with paths provided via command-line arguments.
    # This allows for flexible runtime configuration.
    # The logger is not fully configured yet; messages about overrides will appear after setup_logging.
    if args.data_dir:
        config.data_dir = args.data_dir
    
    # Preserve original default output_dir and log_file to correctly adjust relative paths if output_dir is changed.
    original_output_dir = config.output_dir 
    original_log_file = config.log_file

    if args.output_dir:
        config.output_dir = args.output_dir
        # If checkpoint_dir was using the default (relative to original_output_dir), update it to be relative to the new output_dir.
        if config.checkpoint_dir == original_output_dir / "checkpoints" and not args.checkpoint_dir: # only update if not explicitly set
            config.checkpoint_dir = config.output_dir / "checkpoints"
        # If log_file was using the default (relative to original_output_dir), update it.
        # This will be further overridden if args.log_file is also provided.
        if original_log_file == original_output_dir / "training.log" and not args.log_file: # check against original default
            config.log_file = config.output_dir / "training.log"

    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.cache_dir:
        config.cache_dir = args.cache_dir
    
    # Log file override has the highest precedence for the log file path.
    # If args.log_file is an absolute path, it's used directly.
    # If it's relative, it's made relative to the (potentially overridden and now final) config.output_dir.
    if args.log_file: 
        if args.log_file.is_absolute():
            config.log_file = args.log_file
        else: 
            config.log_file = config.output_dir / args.log_file # Make relative to the final output_dir
    # --- Directory Creation and Logging Setup ---
    # Ensure all necessary directories exist based on the final configuration.
    # The Config class's __init__ also creates these for default paths, but re-running
    # mkdir here ensures that overridden paths are also created.
    # Ensure all necessary directories (data, output, checkpoints, cache, log parent) exist.
    # This is done after all overrides to use the final path values.
    # The Config class's __init__ already does this for default paths, but we re-ensure for overridden paths.
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    config.log_file.parent.mkdir(parents=True, exist_ok=True) 

    # Configure application-wide logging using the determined log file path.
    # This must be done AFTER config overrides, especially for `config.log_file`.
    setup_logging(config.log_file)
    # Get a logger for the main module, now that logging is configured.
    logger = logging.getLogger(__name__) 

    # --- Main Application Logic ---
    try:
        # Log final configuration values and command-line arguments.
        logger.info("Application starting with resolved configuration:")
        for key, value in vars(config).items(): # Log all attributes of the config object
            logger.info(f"  Config - {key}: {value}")
        logger.info("Parsed command-line arguments:")
        for key, value in vars(args).items():
            logger.info(f"  Args - {key}: {value}")

        # --- Device and Performance Configuration ---
        # Configure device (GPU/CPU) and apply A100-specific optimizations if applicable.
        logger.info("Configuring device and performance settings...")
        device = A100Optimizer.configure_device()  # This method handles its own logging
        A100Optimizer.optimize_performance()

        A100Optimizer.optimize_performance() # This method handles its own logging

        # --- Trainer Initialization and Training ---
        # Initialize the trainer with data paths (relative to config.data_dir) and the config object.
        logger.info("Initializing DistributedRadiologyTrainer...")
        trainer = DistributedRadiologyTrainer(
            train_csv=args.train_csv,       # Name of the training CSV file
            val_csv=args.val_csv,         # Name of the validation CSV file
            config=config,                # Pass the fully configured Config object
            device=device                 # Pass the configured PyTorch device
        )
        logger.info("DistributedRadiologyTrainer initialized.")

        # Start the training process for the specified number of epochs.
        # The trainer's .train() method should handle its own internal logging for epochs and steps.
        logger.info(f"Starting training for {args.epochs} epochs...")
        trainer.train(epochs=args.epochs)
        logger.info("Training finished.")

        # --- Evaluation Phase ---
        # If --eval flag is set, perform evaluation using the validation dataset.
        if args.eval:
            logger.info("Starting evaluation phase...")
            # MedicalEvaluator is initialized with the device.
            # It loads its own models (CLIP, CheXbert) as needed.
            evaluator = MedicalEvaluator(device=device) 
            
            # Retrieve validation data (predictions, references, images) from the trainer.
            # This assumes trainer has a method to provide these after training or from a loaded state.
            logger.info("Retrieving validation data from trainer for evaluation...")
            val_predictions, val_references, val_images = trainer.get_validation_data() # This method needs to exist in trainer
            
            if val_predictions and val_references and val_images:
                logger.info(f"Evaluating {len(val_predictions)} predictions against {len(val_references)} references with {len(val_images)} images.")
                eval_results = evaluator.evaluate(
                    predictions=val_predictions,
                    references=val_references,
                    images=val_images # PIL Images expected by MedicalEvaluator
                )
                logger.info(f"Evaluation Results: {eval_results}")
            else:
                logger.warning("Evaluation skipped: No validation data available from trainer.")
        else:
            logger.info("Evaluation phase skipped as --eval flag was not set.")

        # --- Deployment Phase ---
        # If --deploy flag is set, proceed with model deployment steps.
        if args.deploy:
            logger.info("Starting deployment phase...")
            # Find the latest (or best) checkpoint from the configured checkpoint directory.
            checkpoints = sorted(list(config.checkpoint_dir.glob('*.pt'))) 
            if not checkpoints:
                logger.error(f"No checkpoints found in {config.checkpoint_dir}. Deployment aborted.")
            else:
                latest_checkpoint = checkpoints[-1] # Example: using the last saved checkpoint
                logger.info(f"Using checkpoint for deployment: {latest_checkpoint}")
            
                # Path for the deployment-specific configuration (e.g., ONNX export settings).
                deploy_config_path = config.output_dir / 'deploy_config.json'
                logger.info(f"Deployment config path: {deploy_config_path} (Note: this file might be created by deployer or used if exists)")
            
                # Initialize the deployer with the model path, deployment config path, and device.
                deployer = RadiologyDeployer(
                    model_path=str(latest_checkpoint),
                    config_path=str(deploy_config_path),
                    device=device
                )
            
                # Define the output path for the ONNX model, using the configured output directory.
                onnx_model_path = config.output_dir / 'model.onnx'
                logger.info(f"Exporting model to ONNX format at: {onnx_model_path}")
                deployer.export_onnx(str(onnx_model_path)) 
                
                logger.info("Optimizing ONNX model for inference (if applicable in deployer)...")
                deployer.optimize_for_inference() # Further optimizations on the loaded model
                
                logger.info(f"Model deployment tasks completed. Main ONNX model at: {onnx_model_path}")
        else:
            logger.info("Deployment phase skipped as --deploy flag was not set.")

    # --- Error Handling and Cleanup ---
    # Catch specific exceptions for more informative error messages.
    except FileNotFoundError as e:
        logger.error(f"Operation failed: A required file was not found. {e}", exc_info=True)
    except IOError as e:
        logger.error(f"Operation failed: An I/O error occurred (e.g., file read/write). {e}", exc_info=True)
    except RuntimeError as e: # Catch PyTorch runtime errors, e.g., CUDA OOM
        logger.error(f"Operation failed: A runtime error occurred. {e}", exc_info=True)
    except Exception as e: # Catch any other unexpected errors
        logger.critical(f"An unexpected critical error occurred in the main operation: {e}", exc_info=True)
        # Depending on the application, might want to re-raise or exit with error code
    finally:
        # Ensure resources are cleaned up, regardless of success or failure.
        logger.info("Performing final cleanup...")
        if 'trainer' in locals() and trainer is not None: # Check if trainer was initialized
            trainer.cleanup() # Trainer's cleanup should handle its own resources (e.g., wandb)
            
            # Optional: Cleanup for dataset caches if not handled by trainer or if trainer init failed.
            # These were interactive and might be better handled by a separate utility or config flags.
            # if hasattr(trainer, 'train_dataset') and trainer.train_dataset is not None:
            #      if hasattr(trainer.train_dataset, 'cleanup_cache'):
            #         logger.info("Cleaning up training dataset cache (if applicable and non-interactive)...")
            #         # trainer.train_dataset.cleanup_cache(force_delete=True) # Example non-interactive
            # if hasattr(trainer, 'val_dataset') and trainer.val_dataset is not None:
            #     if hasattr(trainer.val_dataset, 'cleanup_cache'):
            #         logger.info("Cleaning up validation dataset cache (if applicable and non-interactive)...")
            #         # trainer.val_dataset.cleanup_cache(force_delete=True) # Example non-interactive
        else:
            logger.info("Trainer was not initialized or already cleaned. No trainer-specific cleanup needed.")
        logger.info("Application shutdown process complete.")

if __name__ == "__main__":
    # The basicConfig for logging is set at the top for very early messages.
    # Full logging setup (to file, etc.) happens inside main() after config parsing.
    main()