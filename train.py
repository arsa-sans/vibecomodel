import os
import torch
import argparse
from src.utils.config import load_yaml
from src.utils.device import get_device
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.utils.helpers import load_checkpoint
from src.data.loader import get_dataloaders
from src.training.trainer import Trainer
from src.training.loss import get_criterion
from src.training.optimizer import get_optimizer
from src.training.scheduler import get_scheduler
from src.model.architecture import get_model_architecture

def main():
    parser = argparse.ArgumentParser(description="Waste Classification Training")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    args = parser.parse_args()
    
    # Load configs
    train_cfg = load_yaml("configs/train.yaml")
    model_cfg = load_yaml("configs/model.yaml")
    data_cfg = load_yaml("configs/dataset.yaml")
    
    # Setup
    logger = setup_logger("train")
    device = get_device()
    set_seed(train_cfg.get('seed', 42))
    
    logger.info(f"Using device: {device}")
    
    # Data
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir=data_cfg['data_path'],
        batch_size=train_cfg['batch_size'],
        img_size=data_cfg['img_size'],
        num_workers=data_cfg['num_workers']
    )
    
    logger.info(f"Loaded {len(class_names)} classes: {class_names}")
    
    # Model
    model = get_model_architecture(
        model_name=model_cfg['name'],
        num_classes=len(class_names),
        pretrained=model_cfg['pretrained']
    ).to(device)
    
    # Training components
    criterion = get_criterion(train_cfg['loss'])
    optimizer = get_optimizer(
        model.parameters(), 
        name=train_cfg['optimizer'], 
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay']
    )
    scheduler = get_scheduler(optimizer, name=train_cfg['scheduler'])
    
    # Resume logic
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        checkpoint_path = os.path.join(train_cfg['output_path'], "latest_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            start_epoch, best_acc = load_checkpoint(checkpoint_path, model, optimizer)
            logger.info(f"Resuming training from epoch {start_epoch} (Best Acc: {best_acc:.2f}%)")
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
            
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=train_cfg,
        logger=logger
    )
    
    # Set best accuracy if resuming
    if args.resume:
        trainer.best_val_acc = best_acc
    
    # Training
    logger.info("Starting training...")
    trainer.fit(num_epochs=train_cfg['epochs'], start_epoch=start_epoch)
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
