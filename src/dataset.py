import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
from pathlib import Path
import shutil
import logging
from typing import Dict, Any

# Set multiprocessing strategy for file system sharing
torch.multiprocessing.set_sharing_strategy('file_system')
logger = logging.getLogger(__name__)
DEBUG = False # Set to True for debugging with a smaller dataset

class MIMICCXRDataset(Dataset):
    """
    A PyTorch Dataset for the MIMIC-CXR dataset.

    This class handles loading images and reports, with support for caching to
    speed up data loading during training. It is designed to be robust against
    common issues like missing files or corrupted cache entries.
    """
    def __init__(self, csv_file: str, transform: Any = None, cache_dir: str = "/tmp/mimic_cache"):
        """
        Args:
            csv_file (str): Path to the CSV file containing image paths and reports.
            transform (Any, optional): Transformations to be applied to the images.
                                       Defaults to a standard set of transformations.
            cache_dir (str, optional): Directory to store cached samples.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.cache_dir.exists():
            raise RuntimeError(f"Failed to create cache directory: {self.cache_dir}")

        try:
            self.data = pd.read_csv(csv_file, nrows=1000 if DEBUG else None)
        except FileNotFoundError:
            logger.error(f"CSV file not found at {csv_file}")
            raise

        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]) # Grayscale normalization
        ])
        
    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a sample from the dataset.
        
        Tries to load from cache first. If not available, it loads the image
        from disk, applies transformations, and saves it to the cache.
        """
        cache_path = self.cache_dir / f"sample_{idx}.pt"
        
        if cache_path.exists():
            try:
                return torch.load(cache_path)
            except Exception as e:
                logger.warning(f"Corrupted cache file {cache_path}, recreating. Error: {e}")
                cache_path.unlink()

        try:
            row = self.data.iloc[idx]
            image_path = row['image_path']
            
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
                
            image = Image.open(image_path).convert('L') # Convert to grayscale
            image = self.transform(image)
            
            report = row['report']
            labels = torch.tensor([float(x.strip()) for x in row['labels'].split(',')])
            
            sample = {'image': image, 'report': report, 'labels': labels}
            torch.save(sample, cache_path)
            
            return sample
        except Exception as e:
            logger.error(f"Error loading sample {idx} ({row.get('image_path', 'N/A')}): {e}")
            return {
                'image': torch.zeros((1, 256, 256)),
                'report': "Error loading report.",
                'labels': torch.zeros(5)
            }
            
    def cleanup_cache(self):
        """
        Removes all cached files from the cache directory.
        This method is non-interactive and will delete files immediately.
        """
        if self.cache_dir.exists():
            try:
                shutil.rmtree(self.cache_dir)
                logger.info(f"Cache directory {self.cache_dir} has been removed.")
                self.cache_dir.mkdir(parents=True, exist_ok=True) # Recreate for future use
            except Exception as e:
                logger.error(f"Failed to remove cache directory {self.cache_dir}: {e}")
