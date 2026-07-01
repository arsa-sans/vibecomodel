import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from src.utils.config import load_yaml
from src.utils.device import get_device
from src.model.model import ModelWrapper
from src.inference.predict import Predictor

def diagnose():
    print("--- Starting API Diagnostics ---")
    
    try:
        print("1. Checking Configs...")
        model_cfg = load_yaml("configs/model.yaml")
        data_cfg = load_yaml("configs/dataset.yaml")
        print(f"   Model Name: {model_cfg.get('name')}")
        print(f"   Data Path: {data_cfg.get('data_path')}")
        
        print("\n2. Checking Device...")
        device = get_device()
        print(f"   Using device: {device}")
        
        print("\n3. Checking Class Names...")
        data_path = data_cfg['data_path']
        if os.path.exists(data_path):
            class_names = sorted(os.listdir(data_path))
            print(f"   Found {len(class_names)} classes.")
        else:
            print(f"   WARNING: Data path {data_path} not found!")
            class_names = ["placeholder"]
            
        print("\n4. Initializing ModelWrapper...")
        model_wrapper = ModelWrapper(
            model_name=model_cfg['name'],
            num_classes=len(class_names),
            device=device
        )
        print("   ModelWrapper initialized.")
        
        print("\n5. Checking Checkpoint...")
        checkpoint_path = "models/checkpoints/best_model.pth"
        if os.path.exists(checkpoint_path):
            print(f"   Checkpoint found at {checkpoint_path}. Loading...")
            model_wrapper.load_weights(checkpoint_path)
            print("   Weights loaded successfully.")
        else:
            print(f"   CRITICAL: Checkpoint {checkpoint_path} NOT FOUND.")
            
        print("\n6. Initializing Predictor...")
        predictor = Predictor(model_wrapper, class_names, img_size=data_cfg['img_size'])
        print("   Predictor initialized.")
        
        print("\n--- Diagnostics Finished: SUCCESS ---")
        return True
        
    except Exception as e:
        print(f"\n--- Diagnostics Finished: FAILED ---")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    diagnose()
