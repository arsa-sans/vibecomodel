import os
import sys

# Fix Windows CP1252 encoding crash from PyTorch emoji log output
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import torch
import onnx
from src.utils.config import load_yaml
from src.model.architecture import get_model_architecture

def main():
    print("Mengekspor model PyTorch ke ONNX...")
    
    # Load configs
    model_cfg = load_yaml("configs/model.yaml")
    print(f"Menggunakan arsitektur model: {model_cfg['name']}")
    
    # 7 classes (alphabetical order as per sorting JSON keys)
    class_names = ['glass', 'hazardous', 'metal', 'organic', 'paper', 'plastic', 'textile']
    num_classes = len(class_names)
    print(f"Kategori sampah ({num_classes} kelas): {class_names}")
    
    # Instantiate the architecture
    # We set pretrained=False because we will load our own trained weights
    model = get_model_architecture(
        model_name=model_cfg['name'],
        num_classes=num_classes,
        pretrained=False
    )
    
    # Load weights
    checkpoint_path = "models/checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"File checkpoint model tidak ditemukan di {checkpoint_path}")
        
    print(f"Memuat checkpoint dari {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    # Set to evaluation mode
    model.eval()
    
    # Create dummy input: batch size 1, 3 channels, 224x224 image
    # Standard image format for MobileNetV3 (RGB channels)
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
    
    # Create output directory
    export_dir = "models/exported"
    os.makedirs(export_dir, exist_ok=True)
    onnx_path = os.path.join(export_dir, "best_model.onnx")
    
    print(f"Menjalankan torch.onnx.export ke {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamo=False,         # Force legacy exporter — PyTorch 2.12 new exporter min opset is 18
    )
    print("Ekspor ONNX selesai!")
    
    # Verify the model
    print("Memverifikasi berkas ONNX yang dihasilkan...")
    onnx_model = onnx.load(onnx_path)
    
    # Pin IR version to 8 — onnxruntime 1.4.1 supports max IR version 9.
    # Newer PyTorch/ONNX exporters may emit IR version 10 even for lower opsets.
    onnx_model.ir_version = 8
    
    onnx.checker.check_model(onnx_model)
    print("Verifikasi ONNX sukses! Model valid.")
    
    # Save as self-contained model
    single_onnx_path = os.path.join(export_dir, "best_model_single.onnx")
    print(f"Menyimpan sebagai file ONNX tunggal di {single_onnx_path}...")
    onnx.save_model(onnx_model, single_onnx_path, save_as_external_data=False)
    
    # Replace original and remove external data
    print("Membersihkan file data eksternal...")
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
    data_path = onnx_path + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)
        
    os.rename(single_onnx_path, onnx_path)
    print("Selesai! Model ONNX tunggal tersimpan di:", onnx_path)

if __name__ == "__main__":
    main()
