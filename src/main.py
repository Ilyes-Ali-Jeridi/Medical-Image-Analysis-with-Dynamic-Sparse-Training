#main.py
import os
import logging
import torch
from pathlib import Path
import argparse
import nltk

# Download NLTK data at startup
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logging.error(f"Failed to download NLTK data: {str(e)}")
    raise

from config import Config
from trainer import DistributedRadiologyTrainer
from optimizer import A100Optimizer
from evaluator import MedicalEvaluator
from deployer import RadiologyDeployer

def setup_logging():
    """Configure logging with proper formatting."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='Medical Image Training')
    parser.add_argument('--train-csv', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--val-csv', type=str, required=True, help='Path to validation CSV')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--eval', action='store_true', help='Run evaluation')
    parser.add_argument('--deploy', action='store_true', help='Deploy model')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--cache-dir', type=str, default='/tmp/mimic_cache', help='Cache directory')
    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Initialize config
        config = Config()
        
        # Configure device and distributed training
        device = A100Optimizer.configure_device()
        A100Optimizer.optimize_performance()

        # Initialize trainer
        trainer = DistributedRadiologyTrainer(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            config=config,
            device=device,
            cache_dir=args.cache_dir
        )

        # Train model
        trainer.train(epochs=args.epochs)

        if args.eval:
            logger.info("Starting evaluation...")
            evaluator = MedicalEvaluator(device)
            val_predictions, val_references, val_images = trainer.get_validation_data()
            eval_results = evaluator.evaluate(
                predictions=val_predictions,
                references=val_references,
                images=val_images
            )
            logger.info(f"Evaluation Results: {eval_results}")

        # Deployment phase
        if args.deploy:
            logger.info("Starting deployment...")
            latest_checkpoint = sorted(Path(args.checkpoint_dir).glob('*.pt'))[-1]
            config_path = Path(args.checkpoint_dir) / 'deploy_config.json'
            
            deployer = RadiologyDeployer(
                model_path=str(latest_checkpoint),
                config_path=str(config_path),
                device=device
            )
            
            deployer.export_onnx('model.onnx')
            deployer.optimize_for_inference()
            logger.info("Model deployed successfully")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        # Cleanup
        trainer.cleanup()
        if 'trainer' in locals():
            trainer.train_dataset.cleanup_cache()
            trainer.val_dataset.cleanup_cache()

if __name__ == "__main__":
    main()