import argparse
import torch
from src.utils.config import load_yaml
from src.utils.device import get_device
from src.model.model import ModelWrapper
from src.inference.predict import Predictor
import os

def main():
    parser = argparse.ArgumentParser(description="Garbage Classification Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="models/checkpoints/best_model.pth", help="Path to model weights")
    args = parser.parse_args()
    
    # Load configs
    model_cfg = load_yaml("configs/model.yaml")
    data_cfg = load_yaml("configs/dataset.yaml")
    
    device = get_device()
    
    # We need class names. In a real scenario, we might save them in a JSON.
    # For now, we'll get them from the data directory if it exists, or use a placeholder.
    # PRO-TIP: Always export class_names with the model.
    data_path = data_cfg['data_path']
    if os.path.exists(data_path):
        class_names = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    else:
        # Fallback: read class names from knowledge base JSON
        import json
        kb_path = os.path.join("src", "utils", "waste_info.json")
        if os.path.exists(kb_path):
            with open(kb_path, 'r') as f:
                class_names = sorted(json.load(f).keys())
            print(f"Data directory not found. Using {len(class_names)} classes from knowledge base.")
        else:
            raise FileNotFoundError(
                f"Data directory '{data_path}' not found and no fallback knowledge base available. "
                f"Please run training first or provide the dataset."
            )
        
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
    
    # Prediction
    result = predictor.predict_image(args.image)
    
    print(f"\n" + "="*50)
    print(f"Prediksi: {result['class'].replace('_', ' ').title()}")
    print(f"Tingkat Kepercayaan: {result['confidence']*100:.2f}%")
    print("-" * 50)
    
    if 'disposal_method' in result:
        print(f"Cara Membuang: {result['disposal_method']}")
        print(f"Ide Daur Ulang: {result['recycling_ideas']}")
        print(f"Dampak Lingkungan: {result['environmental_impact']}")
        if 'notes' in result:
            print(f"Catatan Tambahan: {result['notes']}")
    else:
        print("Detail tambahan tidak tersedia untuk kategori ini.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
