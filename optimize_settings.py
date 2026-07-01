import torch
import multiprocessing
import os
import yaml

def optimize():
    print("--- 🛠️ AI Hardware Optimizer ---")
    
    # 1. Detect Hardware
    cpu_cores = multiprocessing.cpu_count()
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None"
    
    print(f"Detected CPU Cores: {cpu_cores}")
    print(f"Detected GPU: {gpu_name}")
    
    # 2. Recommended Settings
    # Dataset Config
    dataset_cfg_path = "configs/dataset.yaml"
    with open(dataset_cfg_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
        
    data_cfg['num_workers'] = max(1, cpu_cores // 2)
    
    # Train Config
    train_cfg_path = "configs/train.yaml"
    with open(train_cfg_path, 'r') as f:
        train_cfg = yaml.safe_load(f)
        
    if cuda_available:
        train_cfg['use_amp'] = True
        train_cfg['batch_size'] = 32 # Safe default for 4GB+ VRAM
    else:
        train_cfg['use_amp'] = False
        train_cfg['batch_size'] = 16 # Lighter for CPU
        
    # 3. Apply Changes
    with open(dataset_cfg_path, 'w') as f:
        yaml.dump(data_cfg, f)
    with open(train_cfg_path, 'w') as f:
        yaml.dump(train_cfg, f)
        
    print("\n✅ Konfigurasi telah dioptimalkan untuk laptop Anda!")
    print(f"- Num Workers: {data_cfg['num_workers']} (berdasarkan CPU)")
    print(f"- Batch Size: {train_cfg['batch_size']}")
    print(f"- Mixed Precision (AMP): {train_cfg['use_amp']} (berdasarkan GPU)")
    print("\nSiap untuk training dengan akurasi tinggi!")

if __name__ == "__main__":
    try:
        optimize()
    except Exception as e:
        print(f"Gagal melakukan optimasi otomatis: {e}")
        print("Harap instal library yang diperlukan: pip install pyyaml torch")
