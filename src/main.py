"""
Main entry point for the medical imaging system.

This script handles training, evaluation, and deployment of the RAG model
for medical report generation. It ties together the configuration, data loading,
training, and deployment components.
"""
import logging
import torch
from pathlib import Path
import argparse
import nltk

# Attempt to download NLTK data at startup, failing gracefully
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logging.warning(f"Could not download NLTK 'punkt' package: {e}")

from .config import Config
from .trainer import DistributedRadiologyTrainer
from .optimizer import A100Optimizer
from .evaluator import MedicalEvaluator
from .deployer import RadiologyDeployer

def setup_logging():
    """
    Configures logging to output to both a file and the console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('medical_training.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """
    Main function to run the training, evaluation, and deployment pipeline.

    Parses command-line arguments to control the workflow, initializes all
    necessary components, and executes the requested stages.
    """
    parser = argparse.ArgumentParser(description='Medical Image Analysis with Dynamic Sparse Training')
    parser.add_argument('--train-csv', type=str, required=True, help='Path to the training data CSV file.')
    parser.add_argument('--val-csv', type=str, required=True, help='Path to the validation data CSV file.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--eval', action='store_true', help='Run evaluation after training.')
    parser.add_argument('--deploy', action='store_true', help='Run deployment after training.')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save model checkpoints.')
    parser.add_argument('--cache-dir', type=str, default='/tmp/mimic_cache', help='Directory for caching dataset samples.')
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)
    trainer = None

    try:
        config = Config()
        device = A100Optimizer.configure_device()
        A100Optimizer.optimize_performance()

        logger.info("Initializing trainer...")
        trainer = DistributedRadiologyTrainer(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            config=config,
            device=device
        )

        logger.info("Starting training...")
        trainer.train(epochs=args.epochs)
        logger.info("Training complete.")

        if args.eval:
            logger.info("Starting evaluation...")
            evaluator = MedicalEvaluator(device)
            predictions, references, images = trainer.get_validation_data()
            results = evaluator.evaluate(predictions, references, images)
            logger.info(f"Evaluation Results: {results}")

        if args.deploy:
            logger.info("Starting deployment...")
            latest_checkpoint = max(Path(args.checkpoint_dir).glob('*.pt'), key=lambda p: p.stat().st_mtime)
            
            deployer = RadiologyDeployer(
                model_path=str(latest_checkpoint),
                device=device
            )
            
            deployer.export_onnx('vision_encoder.onnx')
            deployer.optimize_for_inference()
            logger.info("Deployment artifacts created successfully.")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise
    finally:
        if trainer:
            logger.info("Cleaning up resources...")
            trainer.cleanup()
            if hasattr(trainer, 'train_dataset'):
                trainer.train_dataset.cleanup_cache()
            if hasattr(trainer, 'val_dataset'):
                trainer.val_dataset.cleanup_cache()
            logger.info("Cleanup complete.")

if __name__ == "__main__":
    main()