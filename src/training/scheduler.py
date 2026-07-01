import torch.optim.lr_scheduler as scheduler

def get_scheduler(optimizer, name: str = "reduce_lr_on_plateau", patience: int = 5):
    """Factory for learning rate schedulers."""
    if name == "reduce_lr_on_plateau":
        return scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience)
    elif name == "cosine_annealing":
        return scheduler.CosineAnnealingLR(optimizer, T_max=10)
    return None
