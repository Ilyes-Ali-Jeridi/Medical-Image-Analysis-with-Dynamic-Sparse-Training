import torch
import onnx
import logging
from typing import Dict, Any

# Assuming models.py and config.py are in the same directory or accessible
from .models import MedicalRAG, MedicalViT
from .config import Config

logger = logging.getLogger(__name__)

class RadiologyDeployer:
    """
    Handles model deployment tasks, including loading, optimization,
    and exporting for inference.
    """
    def __init__(self, model_path: str, device: torch.device):
        """
        Args:
            model_path (str): Path to the trained model checkpoint (.pt file).
            device (torch.device): The device to run the model on.
        """
        self.device = device
        self.model, self.config = self._load_model(model_path)

    def _load_model(self, model_path: str) -> (torch.nn.Module, Config):
        """
        Loads a trained model and its configuration from a checkpoint.
        
        Args:
            model_path (str): The path to the model checkpoint.

        Returns:
            A tuple containing the loaded model and its configuration object.
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            config = checkpoint['config']
            
            vision_encoder = MedicalViT(config)
            model = MedicalRAG(config, vision_encoder=vision_encoder)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            logger.info(f"Model loaded successfully from {model_path}")
            return model, config
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
            
    def export_onnx(self, output_path: str, batch_size: int = 1):
        """
        Exports the vision encoder part of the model to ONNX format.
        
        Args:
            output_path (str): Path to save the ONNX model.
            batch_size (int): The batch size for the dummy input.
        """
        try:
            dummy_input = torch.randn(
                batch_size, 1, self.config.image_size, self.config.image_size,
                device=self.device
            )

            # Export only the vision encoder, as the full RAG model is not ONNX-compatible
            torch.onnx.export(
                self.model.vision_encoder,
                dummy_input,
                output_path,
                opset_version=13,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['features'],
                dynamic_axes={'input': {0: 'batch_size'}, 'features': {0: 'batch_size'}}
            )
            
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            logger.info(f"Vision encoder exported to ONNX: {output_path}")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
            
    def optimize_for_inference(self):
        """
        Applies basic optimizations for inference by freezing model parameters.
        """
        try:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("Model optimized for inference (parameters frozen).")
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            raise
            
    def predict(self, image: torch.Tensor) -> Dict[str, Any]:
        """
        Generates a report for a single input image.

        Args:
            image (torch.Tensor): A single image tensor.

        Returns:
            A dictionary containing the generated report or an error message.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0) # Add batch dimension if missing

        return self.batch_predict(image)
            
    def batch_predict(self, images: torch.Tensor) -> Dict[str, Any]:
        """
        Generates reports for a batch of images.

        Args:
            images (torch.Tensor): A batch of image tensors.

        Returns:
            A dictionary containing the generated reports or an error message.
        """
        try:
            with torch.no_grad():
                outputs = self.model(images.to(self.device))
                return {'reports': outputs['generated_reports'], 'success': True}
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return {'success': False, 'error': str(e)}
