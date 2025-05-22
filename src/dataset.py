import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
import os
from pathlib import Path
import shutil # Added for cleanup_cache
import logging # Added for logging
import pickle # For specific unpickling errors with torch.load
from src.config import Config # Assuming Config is in src/config.py

logger = logging.getLogger(__name__)

# Define DEBUG globally or pass it appropriately if it's meant to be dynamic
# This allows for loading a subset of data for faster iteration during development.
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"


class MIMICCXRDataset(Dataset):
    """
    Dataset class for the MIMIC-CXR dataset.

    Handles loading images and their corresponding reports from a CSV file.
    Implements caching for preprocessed samples to speed up data loading
    during training and evaluation. Includes error handling for missing
    files, corrupted data, and other common dataset issues.

    Attributes:
        data_dir (Path): The base directory where the dataset (e.g., CSV file, images) is located.
        cache_dir (Path): The directory to store and load cached preprocessed samples.
        csv_file_path (Path): The full path to the CSV file containing dataset metadata.
        data (pd.DataFrame): Pandas DataFrame holding the loaded CSV data.
        transform (transforms.Compose): torchvision transforms to be applied to the images.
        prefetch_factor (int): Hint for DataLoader for prefetching data.
    """
    def __init__(self, csv_file: str, data_dir: Path, cache_dir: Path, transform: transforms.Compose = None):
        """
        Initializes the MIMICCXRDataset.

        Args:
            csv_file (str): Name of the CSV file (e.g., "mimic_cxr_data.csv") located in `data_dir`.
            data_dir (Path): Path to the directory containing the dataset files.
            cache_dir (Path): Path to the directory where cached samples will be stored.
            transform (transforms.Compose, optional): torchvision transforms to apply to images.
                If None, a default transform (Resize, ToTensor, Normalize) is used.
        
        Raises:
            RuntimeError: If the cache directory cannot be created.
            FileNotFoundError: If the specified CSV file does not exist in `data_dir`.
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        if not self.cache_dir.exists(): # Double check after creation attempt
            # This should ideally not happen if mkdir was successful, but defensive check
            raise RuntimeError(f"Failed to create or access cache directory: {self.cache_dir}")

        # Construct and validate the full path for the CSV file
        self.csv_file_path = self.data_dir / csv_file
        if not self.csv_file_path.exists():
            logger.error(f"CSV file not found at {self.csv_file_path}")
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")

        logger.info(f"Loading data from CSV: {self.csv_file_path}")
        try:
            # Load data: if DEBUG is True, load only a subset (e.g., 1000 rows) for faster debugging.
            self.data = pd.read_csv(self.csv_file_path, nrows=1000 if DEBUG else None)
            if self.data.empty and not DEBUG: # Check if dataframe is empty when not in DEBUG mode (where 1000 rows might still be empty)
                logger.warning(f"CSV file {self.csv_file_path} loaded successfully but is empty.")
            elif self.data.empty and DEBUG:
                logger.info(f"CSV file {self.csv_file_path} loaded (DEBUG mode, max 1000 rows) but is empty.")

        except pd.errors.EmptyDataError:
            logger.error(f"CSV file {self.csv_file_path} is empty.", exc_info=True)
            raise # Re-raise as this is a critical issue for dataset initialization
        except pd.errors.ParserError:
            logger.error(f"Failed to parse CSV file {self.csv_file_path}. Check for corruption.", exc_info=True)
            raise # Re-raise as this is a critical issue
        except Exception as e: # Catch any other pandas related error during read_csv
            logger.error(f"An unexpected error occurred while reading CSV {self.csv_file_path}: {e}", exc_info=True)
            raise

        logger.info(f"Loaded {len(self.data)} records from CSV.")
        
        # Define default transforms if none are provided
        if transform is None:
            logger.info("No transform provided, using default (Resize, ToTensor, Normalize).")
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)), # Standard size, consider making configurable via Config
                transforms.ToTensor(),          # Converts PIL image to PyTorch tensor
                transforms.Normalize(mean=[0.485], std=[0.229]) # Standard normalization for grayscale
            ])
        else:
            self.transform = transform
            logger.info("Using provided transform.")
            
        # This attribute is used by PyTorch DataLoader if prefetch_factor > 0
        self.prefetch_factor = 2 # Could also be part of Config for more fine-grained control

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Retrieves a sample from the dataset at the given index.

        Attempts to load the sample from cache first. If not found or if the
        cache is corrupted, it loads the image from disk, applies transforms,
        and caches the processed sample.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Dict[str, any]: A dictionary containing the processed image tensor,
            the report text, and the corresponding labels. Returns a placeholder
            error sample if loading fails.
        """
        cache_path = self.cache_dir / f"{idx}.pt"
        
        # Construct path to the cached file for this index
        cache_path = self.cache_dir / f"{idx}.pt"
        
        try:
            # Attempt to load from cache if the file exists
            if cache_path.exists():
                try:
                    # Load tensor data from cache; map_location='cpu' is safer for multiprocessing
                    data = torch.load(cache_path, map_location='cpu')
                    logger.debug(f"Loaded sample {idx} from cache: {cache_path}")
                    return data
                except FileNotFoundError:
                    # This case is unlikely if cache_path.exists() is true but included for robustness
                    logger.error(f"Cache file {cache_path} vanished after existing. This is highly unexpected.", exc_info=True)
                    # Proceed to load from disk
                except (RuntimeError, pickle.UnpicklingError, EOFError, AttributeError) as e: # More specific errors for torch.load
                    logger.warning(f"Corrupted or incompatible cache file {cache_path}, attempting to regenerate: {e}", exc_info=True)
                    try:
                        cache_path.unlink(missing_ok=True) # Remove potentially corrupted cache file
                    except OSError as unlink_e:
                        logger.error(f"Failed to delete corrupted cache file {cache_path}: {unlink_e}", exc_info=True)
                    # Proceed to load from disk and recache

            # If not loaded from cache, load from original data source
            logger.debug(f"Cache miss or regeneration for sample {idx}. Loading from source.")
            row = self.data.iloc[idx] # Get metadata for the sample
            
            # Ensure 'image_path' column exists and has a value
            image_path_in_csv = row.get('image_path')
            if not image_path_in_csv: # Handles None or empty string
                logger.error(f"Missing or empty 'image_path' in CSV for index {idx}. Row data: {row.to_dict()}")
                return self._get_error_sample(f"Missing 'image_path' in CSV for index {idx}")

            # Construct the full image path
            image_path = self.data_dir / image_path_in_csv

            # Check if image file exists
            if not image_path.exists():
                logger.error(f"Image file not found for sample {idx} at {image_path}. CSV reference: {image_path_in_csv}")
                return self._get_error_sample(f"Image not found: {image_path}")

            # Load and process the image
            try:
                image = Image.open(image_path).convert('L') # Open as PIL Image and convert to grayscale
            except FileNotFoundError: # Should be caught by image_path.exists() but as a safeguard
                logger.error(f"Image file {image_path} not found (race condition or external deletion after check).", exc_info=True)
                return self._get_error_sample(f"Image suddenly not found: {image_path}")
            except (Image.DecompressionBombError, Image.UnidentifiedImageError, IOError) as img_e:
                logger.error(f"Error opening or processing image for sample {idx} at {image_path}: {img_e}", exc_info=True)
                return self._get_error_sample(f"Image load/processing error: {image_path}")

            # Apply transformations if any are defined
            if self.transform:
                image = self.transform(image)
            
            # Get report text; default to empty string if 'report' column is missing
            report = row.get('report', "")
            
            # Get labels; default to a string of zeros if 'labels' column is missing
            labels_str = row.get('labels', "0,0,0,0,0") # Example default, adjust as per actual label structure
            try:
                # Convert comma-separated string of labels to a tensor
                labels = torch.tensor([float(x.strip()) for x in labels_str.split(',')])
            except ValueError as ve:
                logger.error(f"Error parsing labels '{labels_str}' for sample {idx}: {ve}", exc_info=True)
                labels = torch.zeros(5) # Fallback to default zero tensor, adjust size as needed
            
            # Assemble the sample dictionary
            sample = {
                'image': image,
                'report': report,
                'labels': labels
            }
            
            # Attempt to save the processed sample to cache
            try:
                torch.save(sample, cache_path)
                logger.debug(f"Saved sample {idx} to cache: {cache_path}")
            except IOError as save_e: # Catch errors during file writing
                logger.error(f"Failed to save sample {idx} to cache {cache_path}: {save_e}", exc_info=True)
            
            return sample
        
        except FileNotFoundError as e: # Should ideally be caught by specific checks above
            logger.error(f"A file was unexpectedly not found while processing sample {idx}: {e}", exc_info=True)
            return self._get_error_sample(f"Unexpected file not found error for sample {idx}")
        except pd.errors.OutOfBounds: # If idx is out of bounds for self.data
            logger.error(f"Index {idx} is out of bounds for the dataset (length {len(self.data)}).", exc_info=True)
            # This is a critical error, usually indicating a problem with DataLoader or __len__
            # Depending on strictness, one might raise an error here.
            return self._get_error_sample(f"Index {idx} out of bounds for dataset")
        except Exception as e: # Catch-all for any other unexpected errors
            # Determine image_path for logging if it was set before an error
            current_image_path_str = "unknown"
            if 'image_path' in locals() and isinstance(image_path, Path):
                current_image_path_str = str(image_path)
            elif 'image_path_in_csv' in locals() and image_path_in_csv:
                 current_image_path_str = str(self.data_dir / image_path_in_csv)

            logger.error(f"Unexpected error loading sample {idx} (image path attempt: {current_image_path_str}): {e}", exc_info=True)
            return self._get_error_sample(f"Unexpected error for sample {idx}")

    def _get_error_sample(self, error_message: str, num_labels: int = 5) -> Dict[str, any]:
        """
        Returns a consistent placeholder/error sample.

        This method is called when an error occurs during the loading or
        processing of a sample in `__getitem__`. It provides a fallback sample
        with zeroed image tensor and an error message in the report.

        Args:
            error_message (str): A description of the error that occurred.
            num_labels (int, optional): The number of labels expected. Defaults to 5.

        Returns:
            Dict[str, any]: A dictionary representing the error sample.
        """
        # Determine image size from transform if possible, else use a default
        img_size = (256, 256) # Default image size
        if self.transform and hasattr(self.transform, 'transforms'):
            for t in self.transform.transforms:
                if isinstance(t, transforms.Resize):
                    # t.size can be int or tuple/list of 2 ints
                    size = t.size 
                    if isinstance(size, int):
                        img_size = (size, size)
                    elif isinstance(size, (tuple, list)) and len(size) == 2:
                        img_size = tuple(size)
                    else: # Could be other Resize format or unexpected type
                        logger.warning(f"Unexpected size format in Resize transform: {size}. Using default {img_size}.")
                    break 
        return {
            'image': torch.zeros(1, *img_size), # Assumes 1 channel (grayscale), then H, W
            'report': f"Error: {error_message}", # Include error message in report
            'labels': torch.zeros(num_labels) 
        }
            
    def cleanup_cache(self, force_delete: bool = False):
        """
        Safely deletes all files within the cache directory.

        This method can operate interactively (requiring user confirmation)
        or non-interactively (if `force_delete` is True). It logs the actions
        taken and any errors encountered.

        Args:
            force_delete (bool, optional): If True, deletes the cache without
                interactive confirmation. Defaults to False.
        """
        logger.info(f"Attempting to cleanup cache directory: {self.cache_dir}")
        
        # Handle interactive confirmation unless force_delete is True
        if not force_delete:
            logger.warning(
                "This is an interactive cache cleanup. "
                "In automated environments, set force_delete=True or remove this call."
            )
            try:
                # Prompt user for confirmation
                confirm = input(
                    f"Are you sure you want to delete all files in cache directory {self.cache_dir}? (y/n) "
                ).lower()
            except RuntimeError: # Handles cases where stdin is not available (e.g., background processes)
                logger.warning(
                    "No standard input available for cache cleanup confirmation. Skipping interactive deletion. "
                    f"To delete cache, manually remove {self.cache_dir} or use force_delete=True."
                )
                return # Exit if confirmation cannot be obtained

            if confirm != 'y':
                logger.info("Cache deletion aborted by user.")
                return

        # Proceed with deletion
        try:
            if self.cache_dir.exists(): # Check if the directory actually exists
                shutil.rmtree(self.cache_dir) # Recursively delete the directory and its contents
                logger.info(f"Cache directory {self.cache_dir} deleted successfully.")
            else:
                logger.info(f"Cache directory {self.cache_dir} does not exist. No deletion needed.")
            
            # Recreate the cache directory after deletion to ensure it's available for future use
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Cache directory {self.cache_dir} (re)created.")
        except OSError as e: # Catch specific OS-level errors (e.g., permission issues)
            logger.error(f"OS error deleting or recreating cache directory {self.cache_dir}: {e}", exc_info=True)
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"Unexpected error during cache cleanup for {self.cache_dir}: {e}", exc_info=True)
