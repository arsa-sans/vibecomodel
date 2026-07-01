import torch.optim as optim

def get_optimizer(model_params, name: str = "adam", lr: float = 0.001, weight_decay: float = 0.0001):
    """Factory for optimizers."""
    if name == "adam":
        return optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
