import torchvision.transforms as T
import torch

def get_preprocessing(img_size: int = 224):
    """Basic preprocessing for classification."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
