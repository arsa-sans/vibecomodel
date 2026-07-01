import torch
from src.data.augmentation import get_val_augmentation
from PIL import Image
import numpy as np
import json
import os
from src.utils.labels import canonical_label, display_label, DEFAULT_CLASS_NAMES


class Predictor:
    """Class to handle single or batch predictions with extended knowledge."""
    def __init__(self, model_wrapper, class_names, img_size=224, knowledge_base_path="src/utils/waste_info.json"):
        self.model_wrapper = model_wrapper
        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self.transform = get_val_augmentation(img_size)

        # Load knowledge base
        self.knowledge = {}
        if os.path.exists(knowledge_base_path):
            with open(knowledge_base_path, 'r') as f:
                self.knowledge = json.load(f)

        self._recent_predictions = []

    def canonical_label(self, raw_label):
        return canonical_label(raw_label)

    def stable_label(self, label, window_size=5):
        normalized_label = self.canonical_label(label)
        if not normalized_label:
            return "unknown"
        self._recent_predictions.append(normalized_label)
        if len(self._recent_predictions) > window_size:
            self._recent_predictions = self._recent_predictions[-window_size:]
        return self.stabilize_prediction(self._recent_predictions, window_size=window_size)

    def stabilize_prediction(self, labels, window_size=5):
        if not labels:
            return "unknown"
        recent = labels[-window_size:]
        if not recent:
            return "unknown"
        from collections import Counter
        counts = Counter(recent)
        majority = counts.most_common(1)[0][0]
        return majority

    def predict_image(self, image_path):
        """Predict class and provide waste intelligence."""
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        transformed = self.transform(image=image_np)["image"]
        input_tensor = transformed.unsqueeze(0).to(self.model_wrapper.device)

        predicted_idx, confidence = self.model_wrapper.predict(input_tensor)
        class_name = self.class_names[predicted_idx.item()]
        normalized_class = self.canonical_label(class_name)
        stable_class = self.stable_label(normalized_class, window_size=3)

        result = {
            "class": stable_class,
            "confidence": confidence.item(),
            "class_idx": predicted_idx.item(),
            "display_label": display_label(stable_class)
        }

        if stable_class in self.knowledge:
            result.update(self.knowledge[stable_class])

        return result
