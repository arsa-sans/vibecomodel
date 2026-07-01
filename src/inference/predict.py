import torch
from src.data.augmentation import get_val_augmentation
from PIL import Image
import numpy as np
import json
import os

class Predictor:
    """Class to handle single or batch predictions with extended knowledge."""
    def __init__(self, model_wrapper, class_names, img_size=224, knowledge_base_path="src/utils/waste_info.json"):
        self.model_wrapper = model_wrapper
        self.class_names = class_names
        self.transform = get_val_augmentation(img_size)
        
        # Load knowledge base
        self.knowledge = {}
        if os.path.exists(knowledge_base_path):
            with open(knowledge_base_path, 'r') as f:
                self.knowledge = json.load(f)
        
    def predict_image(self, image_path):
        """Predict class and provide waste intelligence."""
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # Apply transforms
        transformed = self.transform(image=image_np)["image"]
        input_tensor = transformed.unsqueeze(0).to(self.model_wrapper.device)
        
        # Predict
        predicted_idx, confidence = self.model_wrapper.predict(input_tensor)
        class_name = self.class_names[predicted_idx.item()]
        
        result = {
            "class": class_name,
            "confidence": confidence.item(),
            "class_idx": predicted_idx.item()
        }
        
        # Add intelligence
        if class_name in self.knowledge:
            result.update(self.knowledge[class_name])
        
        return result
