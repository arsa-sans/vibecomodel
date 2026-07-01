import os
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch

class GarbageDatasetWrapper:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform:
            # Albumentations expects numpy array
            import numpy as np
            x = np.array(x)
            x = self.transform(image=x)["image"]
        return x, y
        
    def __len__(self):
        return len(self.dataset)

def get_dataloaders(data_dir, batch_size=32, img_size=224, val_split=0.2, num_workers=4):
    """Create train and validation dataloaders from an ImageFolder structure."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found at {data_dir}")
        
    full_dataset = ImageFolder(data_dir)
    num_classes = len(full_dataset.classes)
    
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    from src.data.augmentation import get_train_augmentation, get_val_augmentation
    
    train_dataset = GarbageDatasetWrapper(train_subset, transform=get_train_augmentation(img_size))
    val_dataset = GarbageDatasetWrapper(val_subset, transform=get_val_augmentation(img_size))
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset.classes
