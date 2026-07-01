import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from datetime import datetime

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 device, config, logger, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.logger = logger
        
        # Paths
        self.checkpoint_dir = config.get('output_path', 'models/checkpoints')
        self.log_dir = config.get('log_dir', 'outputs/logs')
        
        # Metrics
        self.best_val_acc = 0.0
        self.early_stop_counter = 0
        self.early_stop_patience = config.get('early_stop_patience', 10)
        
        # Logging
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'tensorboard'))
        self.csv_logs = []
        
        # Mixed Precision
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            self.scaler.scale(loss).backward()
            
            if self.config.get('grad_clipping', False):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip_max_norm', 1.0))
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/(total/images.size(0)), 'acc': 100.*correct/total})
            
        return running_loss / len(self.train_loader), 100. * correct / total

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        return running_loss / len(self.val_loader), 100. * correct / total

    def fit(self, num_epochs, start_epoch=0):
        for epoch in range(start_epoch, num_epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            
            # CSV Logging
            self.csv_logs.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            
            # Checkpointing
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.early_stop_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.early_stop_counter += 1
                self.save_checkpoint(epoch, is_best=False)
            
            if self.early_stop_counter >= self.early_stop_patience:
                self.logger.info("Early stopping triggered")
                break
                
        self.save_csv_logs()
        self.writer.close()

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint safely."""
        state = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_val_acc,
        }
        
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            if True:  # Always save latest checkpoint every epoch for accurate resume
                filename = "latest_checkpoint.pth"
                temp_filename = filename + ".tmp"
                path = os.path.join(self.checkpoint_dir, filename)
                temp_path = os.path.join(self.checkpoint_dir, temp_filename)
                
                torch.save(state, temp_path)
                os.replace(temp_path, path) # Atomic swap
            
            if is_best:
                best_filename = "best_model.pth"
                best_temp = best_filename + ".tmp"
                best_path = os.path.join(self.checkpoint_dir, best_filename)
                best_temp_path = os.path.join(self.checkpoint_dir, best_temp)
                
                torch.save(state, best_temp_path)
                os.replace(best_temp_path, best_path)
                self.logger.info(f"Model terbaik disimpan di {best_path}")
                
        except Exception as e:
            self.logger.error(f"Gagal menyimpan checkpoint: {e}. Pastikan ruang penyimpanan disk mencukupi.")

    def save_csv_logs(self):
        df = pd.DataFrame(self.csv_logs)
        df.to_csv(os.path.join(self.log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), index=False)
