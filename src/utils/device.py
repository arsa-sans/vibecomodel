import torch

def get_device() -> torch.device:
    """Detect and return the available device (CUDA or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
