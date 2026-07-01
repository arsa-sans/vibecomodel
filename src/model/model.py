import torch
from src.model.architecture import get_model_architecture

class ModelWrapper:
    """Wrapper class for the model to handle loading and forward pass."""
    def __init__(self, model_name: str, num_classes: int, device: torch.device):
        self.device = device
        self.model = get_model_architecture(model_name, num_classes).to(device)
        
    def load_weights(self, path: str):
        """Load weights from a file."""
        checkpoint = torch.load(path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
    def predict(self, x):
        """Perform inference."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        return predicted, confidence
