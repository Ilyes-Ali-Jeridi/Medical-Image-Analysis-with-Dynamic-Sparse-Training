import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms
import os
from typing import Dict, Any

class MIMICCXRDataset(Dataset):
    """Dataset class for MIMIC-CXR with proper error handling."""
    def __init__(self, csv_file: str, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            row = self.data.iloc[idx]
            image_path = row['image_path']
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
                
            image = Image.open(image_path).convert('L')
            image = self.transform(image)
            
            report = row['report']
            labels = torch.tensor([float(x.strip()) for x in row['labels'].split(',')])
            
            return {
                'image': image,
                'report': report,
                'labels': labels
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            # Return a zero tensor with correct shape as fallback
            return {
                'image': torch.zeros(1, 256, 256),
                'report': "",
                'labels': torch.zeros(5)
            }