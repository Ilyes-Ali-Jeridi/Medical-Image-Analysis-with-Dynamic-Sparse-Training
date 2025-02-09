#deployer.py
import torch
import onnx
import logging
from pathlib import Path
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RadiologyDeployer:
    """Handles model deployment and serving for the medical imaging system."""
    def __init__(self, model_path: str, config_path: str, device: torch.device):
        self.device = device
        self.model = self._load_model(model_path)
        self.config = self._load_config(config_path)
        
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load the trained model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model_state = checkpoint['model_state_dict']
            config = checkpoint['config']
            
            # Initialize model with saved config
            from models import MedicalRAG, MedicalViT
            vision_encoder = MedicalViT(sparse_rate=config.sparse_rate)
            model = MedicalRAG(vision_encoder=vision_encoder, sparse_rate=config.sparse_rate)
            model.load_state_dict(model_state)
            model.to(self.device)
            model.eval()
            
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
            
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load deployment configuration."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            raise
            
    def export_onnx(self, output_path: str, batch_size: int = 1):
        """Export model to ONNX format."""
        try:
            dummy_input = torch.randn(batch_size, 1, 256, 256, device=self.device)
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # Verify exported model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            logger.info(f"Model exported to ONNX: {output_path}")
        except Exception as e:
            logger.error(f"ONNX export failed: {str(e)}")
            raise
            
    def optimize_for_inference(self):
        """Apply inference optimizations."""
        try:
            self.model.eval()
            # Freeze model parameters
            for param in self.model.parameters():
                param.requires_grad = False
                
            # Use torch.jit for optimization
            self.model = torch.jit.script(self.model)
            logger.info("Model optimized for inference")
        except Exception as e:
            logger.error(f"Model optimization failed: {str(e)}")
            raise
            
    def predict(self, image: torch.Tensor) -> Dict[str, Any]:
        """Generate prediction for a single image."""
        try:
            with torch.no_grad():
                image = image.to(self.device)
                output = self.model(image)
                return {
                    'report': output,
                    'success': True
                }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    def batch_predict(self, images: torch.Tensor) -> Dict[str, Any]:
        """Generate predictions for a batch of images."""
        try:
            with torch.no_grad():
                images = images.to(self.device)
                outputs = self.model(images)
                return {
                    'reports': outputs,
                    'success': True
                }
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }