# ♻️ Waste Management AI Assistant

Pipeline klasifikasi gambar sampah profesional dengan panduan pembuangan dan informasi daur ulang.

---

## 📋 Fitur Utama
- **Klasifikasi Cerdas**: Identifikasi jenis sampah menggunakan Deep Learning (MobileNet V3 Large).
- **Saran Langsung**: Panduan membuang, mendaur ulang, dan dampak lingkungan.
- **Asisten Real-Time**: Gunakan kamera webcam untuk identifikasi langsung.
- **Training Tangguh**: Mendukung Mixed Precision, Early Stopping, dan resume training.
- **REST API**: Endpoint FastAPI siap pakai untuk integrasi dengan aplikasi lain.

---

## 🗂️ Struktur Project

```
modeleco/
├── api.py               → REST API (FastAPI) untuk inferensi via HTTP
├── train.py             → Script utama training
├── evaluate.py          → Evaluasi model (akurasi, F1, confusion matrix)
├── predict.py           → Prediksi satu gambar via command line
├── camera_predict.py    → Asisten kamera real-time
├── configs/
│   ├── model.yaml       → Konfigurasi arsitektur model
│   ├── dataset.yaml     → Konfigurasi dataset (path, img_size, dll.)
│   └── train.yaml       → Hyperparameter training
├── datasets/            → Folder data gambar (struktur ImageFolder)
├── models/checkpoints/  → Hasil training (best_model.pth, latest_checkpoint.pth)
├── outputs/             → Log, grafik, dan laporan evaluasi
└── src/                 → Source code inti (model, data pipeline, training, dll.)
```

---

## 🚀 Panduan Lengkap

### 1. Prasyarat

- Python 3.10 atau lebih baru
- (Opsional) GPU NVIDIA dengan CUDA untuk training lebih cepat

### 2. Instalasi & Setup Environment

```powershell
# Buka terminal di folder project
# Buat virtual environment baru
python -m venv venv

# Aktifkan virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
.\venv\Scripts\activate.bat

# Install semua dependensi
pip install -r requirements.txt
```

> **Catatan**: Selalu aktifkan `venv` sebelum menjalankan script apapun.

---

### 3. Persiapan Dataset

Dataset harus menggunakan format **ImageFolder** dari PyTorch:

```
datasets/
└── glass/
│   ├── img1.jpg
│   └── img2.jpg
└── plastic/
│   ├── img3.jpg
└── organic/
    └── img4.jpg
```

- Nama subfolder = nama kategori yang akan dipelajari AI.
- Minimal **50–100 gambar** per kategori untuk hasil yang baik.

---

### 4. Konfigurasi (Opsional)

Edit file di `configs/` sebelum training jika perlu:

| File | Parameter Penting | Nilai Saat Ini |
|---|---|---|
| `model.yaml` | `name` (arsitektur model) | `mobilenet_v3_large` |
| `train.yaml` | `epochs`, `batch_size`, `lr` | 100, 32, 0.001 |
| `dataset.yaml` | `data_path`, `img_size` | `datasets`, 224 |
| `train.yaml` | `early_stop_patience` | 10 (berhenti jika 10 epoch tidak ada peningkatan) |

**Model yang tersedia** (ubah di `configs/model.yaml`):

| Model | Kecepatan | Akurasi | Rekomendasi |
|---|---|---|---|
| `mobilenet_v3_large` | ⚡⚡⚡ Cepat | ★★★★ | **Terbaik untuk laptop** |
| `resnet18` | ⚡⚡ Sedang | ★★★★ | Alternatif ringan |
| `resnet50` | ⚡ Lambat | ★★★★★ | Butuh GPU |
| `efficientnet_b0` | ⚡⚡ Sedang | ★★★★★ | GPU disarankan |

---

### 5. Training Model

```powershell
# Pastikan venv sudah diaktifkan
python train.py
```

Training akan menyimpan:
- `models/checkpoints/best_model.pth` → Model terbaik (akurasi tertinggi)
- `models/checkpoints/latest_checkpoint.pth` → Checkpoint setiap 5 epoch

**Melanjutkan Training (Resume):**

Jika training terputus (laptop mati, dll.), lanjutkan dengan:
```powershell
python train.py --resume
```

**Memantau Training via TensorBoard:**
```powershell
tensorboard --logdir outputs/logs/tensorboard
# Buka browser: http://localhost:6006
```

> ⚠️ **Penting**: Jika Anda mengganti model di `model.yaml`, **hapus terlebih dahulu** file `best_model.pth` dan `latest_checkpoint.pth` lama sebelum training ulang, karena bobot antar arsitektur tidak kompatibel.

---

### 6. Evaluasi Model

Setelah training selesai, ukur performa AI:
```powershell
python evaluate.py
```

Output akan tersimpan di:
- `outputs/reports/evaluation_report.json` → Akurasi, F1-Score per kelas
- `outputs/plots/confusion_matrix.png` → Visualisasi confusion matrix

---

### 7. Prediksi Satu Gambar (Command Line)

```powershell
python predict.py --image "path/ke/gambar.jpg"
```

**Contoh output:**
```
==================================================
Prediksi: Plastic
Tingkat Kepercayaan: 94.37%
--------------------------------------------------
Cara Membuang: Masukkan ke tempat sampah plastik
Ide Daur Ulang: Bisa diolah menjadi paving block
Dampak Lingkungan: Membutuhkan 450 tahun untuk terurai
==================================================
```

---

### 8. Menjalankan REST API

API digunakan untuk mengintegrasikan model ke aplikasi web, mobile, atau sistem lain.

```powershell
# Pastikan best_model.pth sudah ada di models/checkpoints/
python api.py
```

API akan berjalan di: **http://localhost:8000**

**Endpoint yang tersedia:**

| Method | Endpoint | Fungsi |
|---|---|---|
| `GET` | `/health` | Cek apakah model sudah terload |
| `POST` | `/predict` | Kirim gambar, dapatkan prediksi |
| `GET` | `/docs` | Dokumentasi API otomatis (Swagger UI) |

**Contoh request menggunakan `curl`:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/ke/test_image.jpg"
```

**Contoh response:**
```json
{
  "success": true,
  "prediction": {
    "label": "Plastic",
    "raw_label": "plastic",
    "confidence": 94.37
  },
  "intelligence": {
    "disposal": "Masukkan ke tempat sampah plastik",
    "recycling": "Bisa diolah menjadi paving block",
    "impact": "Membutuhkan 450 tahun untuk terurai",
    "notes": ""
  }
}
```

---

### 9. Asisten Kamera Real-Time

```powershell
python camera_predict.py
```

- Tekan **`q`** untuk keluar.
- Pastikan `best_model.pth` sudah ada sebelum menjalankan.

---

## ⚠️ Troubleshooting

| Masalah | Solusi |
|---|---|
| `Fatal error in launcher` saat `pip install` | Virtual environment rusak. Hapus folder `venv/` dan buat ulang dengan `python -m venv venv`. |
| `FileNotFoundError: best_model.pth` | Jalankan `python train.py` terlebih dahulu untuk membuat model. |
| `RuntimeError: Error(s) in loading state_dict` | Arsitektur model di `model.yaml` tidak cocok dengan file `.pth` yang tersimpan. Hapus file `.pth` lama dan training ulang. |
| Training sangat lambat di laptop | Ganti ke `mobilenet_v3_large` di `configs/model.yaml` dan kecilkan `batch_size` ke `16` di `configs/train.yaml`. |
| API error `503 Model not loaded` | Cek `models/checkpoints/best_model.pth` sudah ada, dan pastikan `model.yaml` sesuai dengan model yang dilatih. |
