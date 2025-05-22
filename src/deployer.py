#deployer.py
import torch
import onnx
import logging
from pathlib import Path
import json
import pickle # For specific unpickling errors
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RadiologyDeployer:
    """
    Handles model deployment tasks for the medical imaging system.

    This class is responsible for loading a trained model checkpoint,
    exporting it to ONNX format for optimized inference, applying further
    inference optimizations (like JIT scripting), and providing methods
    for generating predictions.

    Attributes:
        device (torch.device): The device (CPU or CUDA) on which the model will run.
        model (torch.nn.Module): The loaded PyTorch model.
        config (Dict[str, Any]): Deployment-specific configuration loaded from a JSON file.
                                 This config is separate from the training config and typically
                                 holds parameters for ONNX export, image size for dummy inputs, etc.
    """
    def __init__(self, model_path: str, config_path: str, device: torch.device):
        """
        Initializes the RadiologyDeployer.

        Args:
            model_path (str): Path to the trained model checkpoint file (.pt).
            config_path (str): Path to the deployment configuration JSON file.
            device (torch.device): The PyTorch device to use for model loading and inference.
        
        Raises:
            FileNotFoundError: If the `model_path` does not exist.
            AttributeError: If the checkpoint is malformed (e.g., missing 'model_state_dict' or 'config').
            Exception: For other errors during model or configuration loading.
        """
        self.device = device
        logger.info(f"Initializing RadiologyDeployer with model_path: {model_path}, config_path: {config_path}, device: {device}")
        
        # Load the trained model from the checkpoint
        self.model = self._load_model(model_path)
        # Load the deployment-specific configuration
        self.config = self._load_config(config_path)
        
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """
        Loads a trained model from a PyTorch checkpoint file.

        The checkpoint is expected to contain 'model_state_dict' and 'config' (the training configuration object).
        The model architecture (`MedicalViT` and `MedicalRAG`) is re-instantiated based on parameters
        from the saved training configuration. If the training configuration or specific architecture
        parameters (e.g., `image_size`, `sparse_rate`) are missing from the checkpoint,
        sensible defaults are used, and warnings are logged.
        This method assumes the model classes are `MedicalViT` and `MedicalRAG` as defined in the
        `.models` module of this application.

        Args:
            model_path (str): Path to the model checkpoint (.pt file).

        Returns:
            torch.nn.Module: The loaded PyTorch model, set to evaluation mode.

        Raises:
            FileNotFoundError: If the `model_path` does not exist.
            AttributeError: If the checkpoint is missing 'model_state_dict'.
            RuntimeError, pickle.UnpicklingError, EOFError, torch.serialization.SerializationError:
                If there are issues deserializing the checkpoint.
            Exception: For any other unexpected errors during model loading.
        """
        logger.info(f"Loading model from checkpoint: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device) # Load checkpoint to specified device
            
            # Validate checkpoint structure
            if 'model_state_dict' not in checkpoint:
                logger.error(f"Checkpoint {model_path} is missing 'model_state_dict'.")
                raise AttributeError("Checkpoint is missing 'model_state_dict'.")
            
            training_config_obj = checkpoint.get('config') # Get training config object, may be None
            if training_config_obj is None:
                logger.warning(f"Checkpoint {model_path} is missing 'config' (training configuration object). "
                               "Model will be initialized with default architecture parameters. "
                               "This may lead to issues if defaults don't match the trained model.")
            
            model_state = checkpoint['model_state_dict']
            
            # Log the source of training_config parameters for clarity
            if training_config_obj: # Check if training_config_obj is not None before trying to access its attributes
                logger.debug(f"Using training configuration from checkpoint: {vars(training_config_obj)}")
            else:
                logger.debug("No training configuration found in checkpoint; using default parameters for model architecture.")

            # Helper to get attribute from training_config_obj or use default, logging if default is used.
            def get_tc_attr(config_obj, attr_name, default_val):
                if config_obj and hasattr(config_obj, attr_name): # Ensure config_obj is not None
                    return getattr(config_obj, attr_name)
                # Log warning only if config_obj itself was present but lacked the attribute.
                if config_obj: 
                    logger.warning(f"Training config in checkpoint missing '{attr_name}'. Using default value: {default_val}.")
                # If config_obj is None, the earlier warning about missing 'config' (if training_config_obj was None) suffices.
                # Or, if it was an optional attribute from a present config_obj.
                elif not config_obj:
                     logger.debug(f"Accessing '{attr_name}' for model init; training_config_obj is None, using default: {default_val}.")
                return default_val

            # Initialize model architecture using parameters from the *training* config.
            # Defaults are provided via get_tc_attr if training_config_obj is None or specific keys are missing.
            from .models import MedicalRAG, MedicalViT # Assumes these are the model classes.

            vision_encoder = MedicalViT(
                image_size=get_tc_attr(training_config_obj, 'image_size', 224),
                patch_size=get_tc_attr(training_config_obj, 'patch_size', 16),
                embed_dim=get_tc_attr(training_config_obj, 'embed_dim', 768),
                num_layers=get_tc_attr(training_config_obj, 'num_layers', 12),
                sparse_rate=get_tc_attr(training_config_obj, 'sparse_rate', 0.4)
            )
            model = MedicalRAG( # Assumes MedicalRAG also uses sparse_rate from the main training_config_obj
                vision_encoder=vision_encoder,
                sparse_rate=get_tc_attr(training_config_obj, 'sparse_rate', 0.4)
            )
            
            model.load_state_dict(model_state)
            model.to(self.device) # Ensure model is on the correct device
            model.eval() # Set model to evaluation mode
            
            logger.info(f"Model loaded successfully from {model_path} and moved to {self.device}")
            return model
        except FileNotFoundError: # Specific exception for file not found
            logger.error(f"Model checkpoint file not found: {model_path}", exc_info=True)
            raise
        except (RuntimeError, pickle.UnpicklingError, EOFError, torch.serialization.SerializationError) as load_e: # More specific load errors
            logger.error(f"Failed to load model from checkpoint {model_path} due to deserialization or file error: {load_e}", exc_info=True)
            raise
        except AttributeError as attr_e: # Catch if essential keys are missing after check
            logger.error(f"Error accessing expected key in checkpoint {model_path} (e.g. 'model_state_dict'): {attr_e}", exc_info=True)
            raise
        except Exception as e: # Catch-all for other unexpected errors
            logger.error(f"An unexpected error occurred while loading model from {model_path}: {e}", exc_info=True)
            raise
            
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Loads deployment-specific configuration from a JSON file (`deploy_config.json`).

        If the specified `config_path` file is not found, is empty, or is corrupted (e.g., invalid JSON),
        this method logs a warning or error and returns a default deployment configuration.
        The default configuration ensures that essential keys like 'image_size', 'onnx_opset_version',
        'onnx_input_names', and 'onnx_output_names' are present.
        If the loaded configuration is missing any of these essential keys, defaults for those
        specific keys are used and a warning is logged.

        Args:
            config_path (str): Path to the deployment configuration JSON file.

        Returns:
            Dict[str, Any]: A dictionary containing deployment settings, merged with defaults.
        """
        logger.info(f"Loading deployment configuration from: {config_path}")
        
        # Define comprehensive default settings for deployment.
        # These ensure the application has fallback values for critical ONNX export parameters.
        default_config = {
            "image_size": 224,        # Default image size for ONNX dummy input.
            "onnx_opset_version": 13,  # Default ONNX opset version. PyTorch 1.11+ often supports up to 14/15.
                                      # Check compatibility: https://pytorch.org/docs/stable/onnx.html#exporting-a-model-to-onnx
            "onnx_input_names": ["input"], # Default input name for the ONNX model.
            "onnx_output_names": ["output"],# Default output name for the ONNX model.
            # Add other potential deployment-specific keys here if needed (e.g., for quantization, specific providers)
        }
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Deployment config file not found: {config_path}. Using default deployment settings: {default_config}")
                return default_config
            
            with config_file.open('r') as f:
                try:
                    loaded_config_from_file = json.load(f) # Renamed to avoid confusion with outer scope
                    if not isinstance(loaded_config_from_file, dict): # Ensure it's a dictionary
                        logger.error(f"Deployment config file {config_path} does not contain a valid JSON object. Using default settings.")
                        return default_config
                    if not loaded_config_from_file: # Handles empty JSON object {}
                         logger.warning(f"Deployment config file {config_path} is empty. Using default settings.")
                         # Fall through to merge with defaults, effectively using all defaults.
                except json.JSONDecodeError as json_e: # Handle corrupted JSON
                    logger.error(f"Failed to decode JSON from deployment config {config_path}: {json_e}. Using default settings: {default_config}", exc_info=True)
                    return default_config
            
            # Merge loaded config with defaults. Values from loaded_config_from_file override defaults if keys exist.
            merged_config = {**default_config, **loaded_config_from_file}

            # Log if any essential keys were missing from the loaded_config_from_file and are now using defaults.
            for key in default_config: # Iterate over keys that MUST have a default
                if key not in loaded_config_from_file: # Check if the key was present in the file
                    logger.warning(f"Deployment config '{config_path}' was missing key '{key}'. Using default value: {default_config[key]}.")
            
            logger.info(f"Deployment configuration loaded from {config_path} and merged with defaults: {merged_config}")
            return merged_config
        
        except IOError as io_e: # Handles file read errors not covered by Path.exists()
            logger.error(f"IOError when attempting to read deployment config {config_path}: {io_e}. Using default settings: {default_config}", exc_info=True)
            return default_config
        except Exception as e: # Catch-all for other unexpected errors during config loading
            logger.error(f"Unexpected error loading deployment config {config_path}: {e}. Using default settings: {default_config}", exc_info=True)
            return default_config

    def export_onnx(self, output_path: str, batch_size: int = 1):
        """
        Exports the loaded model to ONNX (Open Neural Network Exchange) format.

        The ONNX model is saved to the specified `output_path`. Dynamic axes are set for
        inputs and outputs to allow for variable batch sizes and potentially other dimensions
        (e.g., image height and width, if the model architecture supports it).
        
        Key parameters for export like `opset_version`, `image_size` (for dummy input),
        `input_names`, and `output_names` are sourced from the deployment configuration
        (`self.config`, loaded from `deploy_config.json`). It's recommended to use an
        `opset_version` compatible with the target ONNX runtime environment (e.g., version 13-15
        are common for recent PyTorch versions, but check PyTorch documentation for specifics).
        Consult the PyTorch ONNX export documentation for opset compatibility:
        https://pytorch.org/docs/stable/onnx.html#exporting-a-model-to-onnx

        Args:
            output_path (str): The file path where the ONNX model will be saved.
            batch_size (int, optional): The batch size for the dummy input tensor used during
                                        the ONNX export process. Defaults to 1.
        
        Raises:
            onnx.checker.ValidationError: If the exported ONNX model fails validation.
            RuntimeError, torch.onnx.CheckerError: For errors during the `torch.onnx.export` process.
            IOError: For file system errors when attempting to save the ONNX model.
            Exception: For any other unexpected errors during the export.
        """
        logger.info(f"Starting ONNX export of the model to: {output_path}")
        try:
            output_file = Path(output_path)
            # Ensure the parent directory for the output ONNX file exists, creating it if necessary.
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Retrieve ONNX export parameters from the deployment configuration.
            # These should have defaults set by _load_config if not present in the JSON.
            image_size = self.config.get("image_size") 
            opset_version = self.config.get("onnx_opset_version")
            input_names = self.config.get("onnx_input_names")
            output_names = self.config.get("onnx_output_names")

            logger.info(f"ONNX export parameters: image_size={image_size}, opset_version={opset_version}, "
                        f"input_names={input_names}, output_names={output_names}, export_batch_size={batch_size}")

            # Define dynamic axes for flexible input/output shapes.
            # This allows the ONNX model to accept inputs of varying batch sizes, heights, and widths.
            dynamic_axes = {}
            for name in input_names: # Assuming NCHW format for image inputs
                dynamic_axes[name] = {0: 'batch_size', 2: 'height', 3: 'width'} 
            for name in output_names: # Assuming output batch size is dynamic
                dynamic_axes[name] = {0: 'batch_size'} 
            logger.debug(f"Using dynamic axes for ONNX export: {dynamic_axes}")

            # Create a dummy input tensor required for tracing the model during ONNX export.
            # This input should match the expected input shape and type of the model.
            # Assuming a single image input, grayscale (1 channel).
            dummy_input = torch.randn(
                batch_size,    # Batch size for the dummy input
                1,             # Number of channels (e.g., 1 for grayscale, 3 for RGB)
                image_size,    # Height of the image
                image_size,    # Width of the image
                device=self.device # Ensure dummy input is on the same device as the model
            )
            
            self.model.eval() # Critical: ensure the model is in evaluation mode for consistent export.

            logger.info(f"Exporting model to ONNX format with opset version {opset_version}...")
            torch.onnx.export(
                self.model,               # The PyTorch model to be exported.
                dummy_input,             # A dummy input tensor for tracing.
                str(output_file),        # Path to save the ONNX model (must be a string).
                opset_version=opset_version, # ONNX opset version to use.
                do_constant_folding=True, # Apply constant folding optimization during export.
                input_names=input_names,   # Names for the input nodes in the ONNX graph.
                output_names=output_names, # Names for the output nodes in the ONNX graph.
                dynamic_axes=dynamic_axes  # Specifies dynamic axes for inputs/outputs.
            )
            
            # Verify the integrity and correctness of the exported ONNX model.
            logger.info("Verifying exported ONNX model structure and metadata...")
            onnx_model = onnx.load(str(output_file)) # Load the saved ONNX model.
            onnx.checker.check_model(onnx_model) # This will raise an error if the model is invalid.
            logger.info(f"Model successfully exported to ONNX and verified: {output_file}")

        except (onnx.checker.ValidationError, onnx.onnx_cpp2py_export.checker.ValidationError) as onnx_val_e:
            logger.error(f"ONNX model validation failed for {output_path}: {onnx_val_e}", exc_info=True)
            raise
        except (RuntimeError, torch.onnx.CheckerError) as export_e: # Includes errors from torch.onnx.export itself
            logger.error(f"ONNX export process failed for {output_path} during torch.onnx.export: {export_e}", exc_info=True)
            raise
        except IOError as io_e: # For file system related errors (e.g., permission denied)
             logger.error(f"IOError during ONNX export (e.g., saving file) for {output_path}: {io_e}", exc_info=True)
             raise
        except Exception as e: # Catch any other unexpected error during the export process
            logger.error(f"An unexpected error occurred during ONNX export for {output_path}: {e}", exc_info=True)
            raise
            
    def optimize_for_inference(self):
        """
        Applies inference optimizations to the loaded model.

        Currently, this involves setting the model to evaluation mode,
        disabling gradients for all parameters, and applying TorchScript JIT compilation.
        
        Raises:
            RuntimeError: If TorchScript JIT compilation fails.
            Exception: For any other unexpected errors during optimization.
        """
        logger.info("Optimizing model for inference...")
        try:
            self.model.eval() # Ensure model is in evaluation mode
            
            # Freeze model parameters to disable gradient calculations
            logger.debug("Freezing model parameters (requires_grad=False).")
            for param in self.model.parameters():
                param.requires_grad = False
                
            # Apply TorchScript JIT compilation for potential performance improvements
            # Note: JIT scripting might not support all Python features or model structures.
            logger.info("Applying TorchScript JIT compilation (torch.jit.script)...")
            self.model = torch.jit.script(self.model)
            logger.info("Model successfully JIT-scripted and optimized for inference.")
        except RuntimeError as jit_e:
            logger.error(f"Torch JIT scripting failed during model optimization: {jit_e}", exc_info=True)
            # Model remains the original (non-JITed) version if scripting fails.
            # Depending on requirements, might raise or just log and continue with non-JITed model.
            raise
        except Exception as e:
            logger.error(f"Unexpected error during model optimization: {e}", exc_info=True)
            raise
            
    def predict(self, image: torch.Tensor) -> Dict[str, Any]:
        """
        Generates a prediction (e.g., a medical report) for a single input image.

        Args:
            image (torch.Tensor): A single image tensor, preprocessed and on the correct device.

        Returns:
            Dict[str, Any]: A dictionary containing the generated 'report' and a 'success' status.
                            In case of an error, 'success' is False and 'error' field contains error message.
        """
        logger.debug(f"Received single image for prediction. Shape: {image.shape}, Device: {image.device}")
        try:
            with torch.no_grad(): # Ensure no gradients are computed during inference
                # Move image to the model's device if not already there
                image = image.to(self.device)
                # Model forward pass
                output = self.model(image) 
                # Assuming output is directly the report or a dict containing it
                # Adapt based on actual model output structure
                report_content = output if isinstance(output, str) else output.get('generated_reports', 'Error: Report not found in model output')
                if isinstance(report_content, list): report_content = report_content[0] # Take first if list

                logger.debug(f"Prediction successful for single image. Report: {report_content[:100]}...") # Log snippet
                return {
                    'report': report_content,
                    'success': True
                }
        except RuntimeError as rt_e:
            logger.error(f"Runtime error during single image prediction: {rt_e}", exc_info=True)
            return {'success': False, 'error': str(rt_e)}
        except Exception as e:
            logger.error(f"Unexpected error during single image prediction: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
            
    def batch_predict(self, images: torch.Tensor) -> Dict[str, Any]:
        """
        Generates predictions for a batch of input images.

        Args:
            images (torch.Tensor): A batch of image tensors, preprocessed and on the correct device.

        Returns:
            Dict[str, Any]: A dictionary containing a list of 'reports' and a 'success' status.
                            In case of an error, 'success' is False and 'error' field contains error message.
        """
        logger.debug(f"Received batch of {images.shape[0]} images for prediction. Shape: {images.shape}, Device: {images.device}")
        try:
            with torch.no_grad(): # Ensure no gradients are computed
                # Move images to the model's device
                images = images.to(self.device)
                # Model forward pass for the batch
                outputs = self.model(images)
                # Assuming outputs is a list of reports or a dict containing it
                # Adapt based on actual model output structure
                reports_content = outputs if isinstance(outputs, list) else outputs.get('generated_reports', ['Error: Reports not found in model output'] * images.shape[0])

                logger.debug(f"Batch prediction successful for {images.shape[0]} images.")
                return {
                    'reports': reports_content,
                    'success': True
                }
        except RuntimeError as rt_e:
            logger.error(f"Runtime error during batch prediction: {rt_e}", exc_info=True)
            return {'success': False, 'error': str(rt_e)}
        except Exception as e:
            logger.error(f"Unexpected error during batch prediction: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
