import torch
import numpy as np
import onnxruntime as ort
import os

# Fix encoding
os.environ["PYTHONIOENCODING"] = "utf-8"

from src.utils.config import load_yaml
from src.model.architecture import get_model_architecture

def main():
    model_cfg = load_yaml("configs/model.yaml")
    class_names = ['glass', 'hazardous', 'metal', 'organic', 'paper', 'plastic', 'textile']
    num_classes = len(class_names)
    
    # Load ONNX model
    print("Loading ONNX model...")
    onnx_path = "models/exported/best_model.onnx"
    ort_session = ort.InferenceSession(onnx_path)
    
    # Create zeros input
    dummy_np = np.zeros((1, 3, 224, 224), dtype=np.float32)
    
    # Run ONNX Runtime
    onnx_inputs = {ort_session.get_inputs()[0].name: dummy_np}
    onnx_out = ort_session.run(None, onnx_inputs)[0]
    
    print("\nONNX Logits for Zeros:", onnx_out[0].tolist())

if __name__ == "__main__":
    main()
