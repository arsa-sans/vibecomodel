from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np

def calculate_metrics(y_true, y_pred, labels=None):
    """Calculate comprehensive classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'report': report
    }
