import argparse
import os
from src.utils.config import load_yaml
from src.utils.device import get_device
from src.model.model import ModelWrapper
from src.inference.predict import Predictor
from src.inference.camera import CameraAssistant

def main():
    parser = argparse.ArgumentParser(description="Waste Assistant Camera")
    parser.add_argument("--model", type=str, default="models/checkpoints/best_model.pth", help="Path to model weights")
    parser.add_argument("--cam", type=int, default=0, help="Camera ID")
    args = parser.parse_args()
    
    # Load configs
    model_cfg = load_yaml("configs/model.yaml")
    data_cfg = load_yaml("configs/dataset.yaml")
    
    device = get_device()
    
    # Get class names
    data_path = data_cfg['data_path']
    if os.path.exists(data_path):
        class_names = sorted(os.listdir(data_path))
    else:
        class_names = [f"Class_{i}" for i in range(100)] # Fallback
        
    # Model
    model_wrapper = ModelWrapper(
        model_name=model_cfg['name'],
        num_classes=len(class_names),
        device=device
    )
    
    if os.path.exists(args.model):
        model_wrapper.load_weights(args.model)
        
    # Predictor
    predictor = Predictor(model_wrapper, class_names, img_size=data_cfg['img_size'])
    
    # Assistant
    assistant = CameraAssistant(predictor, camera_id=args.cam)
    assistant.run()

if __name__ == "__main__":
    main()
