import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms
import os
from typing import Dict, Any
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class MIMICCXRDataset(Dataset):
    """Dataset class for MIMIC-CXR with proper error handling and caching."""
    def __init__(self, csv_file: str, transform=None, cache_dir="/tmp/mimic_cache"):
        self.data = pd.read_csv(csv_file)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        self.prefetch_factor = 2
        
    def __len__(self) -> int:
        return len(self.data)
    
    class MIMICCXRDataset(Dataset):
    def __init__(self, csv_file, transform=None, cache_dir="/tmp/mimic_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Safety check for cache directory
        if not self.cache_dir.exists():
            raise RuntimeError(f"Failed to create cache directory: {self.cache_dir}")
            
        # Load data with progress bar
        self.data = pd.read_csv(csv_file, nrows=1000) if DEBUG else pd.read_csv(csv_file)
        
    def __getitem__(self, idx):
        cache_path = self.cache_dir / f"{idx}.pt"
        if cache_path.exists():
            try:
                data = torch.load(cache_path, map_location='cpu')  # Prevent GPU OOM
                data['image'] = data['image'].to(torch.device('cuda'))
                return data
            except Exception as e:
                print(f"Corrupted cache file {cache_path}: {e}")
            
            if cache_path.exists():
                return torch.load(cache_path)
                
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
                
            image = Image.open(image_path).convert('L')
            image = self.transform(image)
            
            report = row['report']
            labels = torch.tensor([float(x.strip()) for x in row['labels'].split(',')])
            
            sample = {
                'image': image,
                'report': report,
                'labels': labels
            }
            
            torch.save(sample, cache_path)
            return sample
            
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            return {
                'image': torch.zeros(1, 256, 256),
                'report': "",
                'labels': torch.zeros(5)
            }
            
    def cleanup_cache(self):
        """Safe cache deletion with confirmation"""
        if input("Delete all cached files? (y/n) ").lower() == 'y':
            shutil.rmtree(self.cache_dir)
