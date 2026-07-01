import numpy as np
import onnxruntime as ort
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os

# Fix encoding
os.environ["PYTHONIOENCODING"] = "utf-8"

def main():
    image_path = "../ecovision_app/assets/test_image.jpg"
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}!")
        return
        
    print("Loading image with PIL...")
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    print(f"PIL Image shape: {image_np.shape}, dtype: {image_np.dtype}")
    
    # Preprocessing transforms (Resize & Normalize)
    transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transform(image=image_np)["image"]
    flat_tensor = transformed.numpy().flatten()
    
    print("\n--- Python Preprocessing Results ---")
    print(f"Preprocessed flat size: {flat_tensor.size}")
    print(f"First 10 values: {flat_tensor[:10].tolist()}")
    print(f"Sum of values: {np.sum(flat_tensor):.6f}")
    print(f"Mean value: {np.mean(flat_tensor):.6f}")
    print(f"Min value: {np.min(flat_tensor):.6f}")
    print(f"Max value: {np.max(flat_tensor):.6f}")
    
    # Run inference
    print("\nRunning inference in Python...")
    onnx_path = "models/exported/best_model.onnx"
    ort_session = ort.InferenceSession(onnx_path)
    onnx_inputs = {ort_session.get_inputs()[0].name: transformed.unsqueeze(0).numpy()}
    onnx_out = ort_session.run(None, onnx_inputs)[0]
    
    # Softmax
    exp_out = np.exp(onnx_out - np.max(onnx_out, axis=1, keepdims=True))
    probs = exp_out / np.sum(exp_out, axis=1, keepdims=True)
    
    class_names = ['glass', 'hazardous', 'metal', 'organic', 'paper', 'plastic', 'textile']
    pred_idx = np.argmax(probs[0])
    print(f"Raw logits: {onnx_out[0].tolist()}")
    print(f"Probabilities: {probs[0].tolist()}")
    print(f"Predicted class: {class_names[pred_idx]} ({probs[0][pred_idx]*100:.2f}%)")

if __name__ == "__main__":
    main()
