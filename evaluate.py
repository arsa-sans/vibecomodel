import torch
from src.utils.config import load_yaml
from src.utils.device import get_device
from src.utils.logger import setup_logger
from src.data.loader import get_dataloaders
from src.model.architecture import get_model_architecture
from src.evaluation.metrics import calculate_metrics
from src.evaluation.confusion_matrix import plot_confusion_matrix
import json
import os

def main():
    # Load configs
    train_cfg = load_yaml("configs/train.yaml")
    model_cfg = load_yaml("configs/model.yaml")
    data_cfg = load_yaml("configs/dataset.yaml")
    
    logger = setup_logger("evaluate")
    device = get_device()
    
    # Data
    _, val_loader, class_names = get_dataloaders(
        data_dir=data_cfg['data_path'],
        batch_size=train_cfg['batch_size'],
        img_size=data_cfg['img_size']
    )
    
    # Model
    model = get_model_architecture(
        model_name=model_cfg['name'],
        num_classes=len(class_names)
    ).to(device)
    
    # Load best weights
    best_model_path = os.path.join(train_cfg['output_path'], "best_model.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best weights from {best_model_path}")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    logger.info("Evaluating on validation set...")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics
    metrics = calculate_metrics(all_labels, all_preds, labels=class_names)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    
    # Save Report
    report_path = "outputs/reports/evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Report saved at {report_path}")
    
    # Confusion Matrix
    cm_path = plot_confusion_matrix(all_labels, all_preds, labels=class_names)
    logger.info(f"Confusion matrix saved at {cm_path}")

if __name__ == "__main__":
    main()
