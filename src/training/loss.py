import torch.nn as nn

def get_criterion(name: str = "cross_entropy"):
    """Factory for loss functions."""
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss: {name}")
