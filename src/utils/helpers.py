import os
import torch

def save_checkpoint(state, checkpoint_dir, filename="checkpoint.pth"):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    return path

def load_checkpoint(path, model, optimizer=None):
    """Load model checkpoint."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_acc', 0.0)
