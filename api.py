import os
import io
import sys
import uuid
import socket
import tempfile
from contextlib import asynccontextmanager

try:
    import torch
except Exception as exc:
    torch = None
    TORCH_IMPORT_ERROR = f"PyTorch import failed: {type(exc).__name__}: {exc}"
else:
    TORCH_IMPORT_ERROR = None

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Global variables
predictor = None
load_error = None

def get_default_class_names():
    return ["glass", "hazardous", "metal", "organic", "paper", "plastic", "textile"]


def get_class_names_from_data_path(data_path: str):
    if not os.path.exists(data_path):
        return get_default_class_names()
    detected_dirs = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    if detected_dirs:
        return detected_dirs
    return get_default_class_names()


def get_available_port(start_port: int = 8000, max_attempts: int = 20) -> int:
    requested_port = int(os.getenv("PORT", start_port))
    if requested_port == 8000:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", 8000))
                return 8000
            except OSError:
                pass
    candidates = [requested_port] + [requested_port + i for i in range(1, max_attempts)]
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Unable to find an available port in range {candidates[:3]}..{candidates[-1]}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, load_error
    print(f"Loading model environment from {BASE_DIR}...")
    try:
        if torch is None:
            raise RuntimeError(TORCH_IMPORT_ERROR or "PyTorch is not available")

        from src.utils.config import load_yaml
        from src.utils.device import get_device
        from src.model.model import ModelWrapper
        from src.inference.predict import Predictor
        
        config_path = os.path.join(BASE_DIR, "configs")
        model_cfg = load_yaml(os.path.join(config_path, "model.yaml"))
        data_cfg = load_yaml(os.path.join(config_path, "dataset.yaml"))
        
        device = get_device()
        
        # Get data path from config and resolve to absolute
        data_path = data_cfg['data_path']
        if not os.path.isabs(data_path):
            data_path = os.path.join(BASE_DIR, data_path)
            
        print(f"Data path: {data_path}")
        
        if os.path.exists(data_path):
            class_names = get_class_names_from_data_path(data_path)
            if set(class_names) != set(get_default_class_names()):
                print(f"Warning: Dataset class folders differ from the expected seven categories. Using discovered order: {class_names}")
        else:
            print(f"Warning: {data_path} not found. Using fallback classes.")
            class_names = get_default_class_names()

        model_wrapper = ModelWrapper(
            model_name=model_cfg['name'],
            num_classes=len(class_names),
            device=device
        )
        
        checkpoint_path = os.path.join(BASE_DIR, "models", "checkpoints", "best_model.pth")
        if os.path.exists(checkpoint_path):
            model_wrapper.load_weights(checkpoint_path)
            print(f"Weights loaded successfully.")
        else:
            print(f"WARNING: Checkpoint {checkpoint_path} not found.")
            
        predictor = Predictor(model_wrapper, class_names, img_size=data_cfg['img_size'])
        print("API is ready for inference.")
    except Exception as e:
        import traceback
        load_error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"CRITICAL ERROR: {load_error}")
    yield

app = FastAPI(
    title="Waste Classification AI API",
    lifespan=lifespan
)

@app.get("/", response_class=HTMLResponse)
def root():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Waste Classification AI</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
                color: #1b5e20;
            }
            .container {
                max-width: 1000px;
                margin: 40px auto;
                background: white;
                border-radius: 16px;
                padding: 32px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.12);
            }
            h1 { margin-top: 0; }
            .chip {
                display: inline-block;
                padding: 8px 12px;
                margin: 6px 6px 0 0;
                background: #2e7d32;
                color: white;
                border-radius: 999px;
                font-size: 14px;
            }
            .card {
                border: 1px solid #dcedc8;
                border-radius: 12px;
                padding: 16px;
                margin-top: 16px;
                background: #f9fff5;
            }
            input[type=file] {
                margin-top: 10px;
                padding: 8px;
            }
            button {
                margin-top: 12px;
                padding: 10px 16px;
                border: none;
                border-radius: 8px;
                background: #2e7d32;
                color: white;
                cursor: pointer;
                margin-right: 8px;
            }
            button.secondary {
                background: #6c757d;
            }
            .preview-wrap {
                display: flex;
                flex-wrap: wrap;
                gap: 16px;
                align-items: flex-start;
            }
            .video-shell {
                position: relative;
                display: inline-block;
                width: 100%;
                max-width: 480px;
                overflow: hidden;
            }
            .video-shell video {
                width: 100%;
                max-width: 480px;
                border-radius: 12px;
                display: block;
                position: relative;
                z-index: 1;
                object-fit: cover;
            }
            .video-shell canvas {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                border-radius: 12px;
                pointer-events: none;
                background: transparent;
                z-index: 2;
            }
            #result {
                margin-top: 16px;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .status {
                margin-top: 8px;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>♻️ Waste Classification AI</h1>
            <p>Unggah gambar atau gunakan kamera untuk melihat prediksi kategori sampah secara langsung.</p>
            <div>
                <span class="chip">paper</span>
                <span class="chip">metal</span>
                <span class="chip">organic</span>
                <span class="chip">hazardous</span>
                <span class="chip">textile</span>
                <span class="chip">glass</span>
                <span class="chip">plastic</span>
            </div>

            <div class="card">
                <h3>1. Upload Gambar</h3>
                <input type="file" id="fileInput" accept="image/*" />
                <br />
                <button onclick="uploadImage()">Prediksi Gambar</button>
                <div id="result">Belum ada hasil.</div>
            </div>

            <div class="card">
                <h3>2. Kamera Langsung</h3>
                <div class="preview-wrap">
                    <div>
                        <div class="video-shell">
                            <video id="video" autoplay playsinline muted></video>
                            <canvas id="overlayCanvas"></canvas>
                        </div>
                        <div class="status" id="cameraStatus">Kamera belum aktif.</div>
                        <div>
                            <button onclick="startCamera()">Buka Kamera</button>
                            <button class="secondary" onclick="stopCamera()">Matikan Kamera</button>
                        </div>
                    </div>
                    <div style="flex:1; min-width:260px;">
                        <h4>Hasil Kamera</h4>
                        <div id="cameraResult">Belum ada prediksi kamera.</div>
                    </div>
                </div>
            </div>
        </div>
        <script>
            const video = document.getElementById('video');
            const overlayCanvas = document.getElementById('overlayCanvas');
            const cameraStatus = document.getElementById('cameraStatus');
            const cameraResult = document.getElementById('cameraResult');
            let stream = null;
            let predictionTimer = null;
            let currentLabel = 'Menunggu...';

            async function uploadImage() {
                const input = document.getElementById('fileInput');
                const result = document.getElementById('result');
                if (!input.files || !input.files[0]) {
                    result.textContent = 'Pilih gambar terlebih dahulu.';
                    return;
                }

                const formData = new FormData();
                formData.append('file', input.files[0]);

                result.textContent = 'Memproses gambar...';
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    if (!response.ok) {
                        throw new Error(data.detail || 'Gagal memprediksi gambar');
                    }
                    result.textContent = JSON.stringify(data, null, 2);
                } catch (error) {
                    result.textContent = 'Error: ' + error.message;
                }
            }

            function setOverlaySize() {
                const width = video.videoWidth || video.clientWidth || 320;
                const height = video.videoHeight || video.clientHeight || 240;
                overlayCanvas.width = width;
                overlayCanvas.height = height;
                overlayCanvas.style.width = `${video.clientWidth}px`;
                overlayCanvas.style.height = `${video.clientHeight}px`;
            }

            async function startCamera() {
                if (stream) return;
                try {
                    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
                    video.srcObject = stream;
                    await video.play();
                    setOverlaySize();
                    cameraStatus.textContent = 'Kamera aktif — memantau objek...';
                    overlayCanvas.hidden = false;
                    video.addEventListener('loadedmetadata', setOverlaySize);
                    window.addEventListener('resize', setOverlaySize);
                    startPredictionLoop();
                } catch (error) {
                    cameraStatus.textContent = 'Tidak bisa mengakses kamera: ' + error.message;
                }
            }

            function stopCamera() {
                if (predictionTimer) {
                    clearInterval(predictionTimer);
                    predictionTimer = null;
                }
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    stream = null;
                }
                if (video.srcObject) {
                    video.srcObject = null;
                }
                overlayCanvas.hidden = true;
                cameraStatus.textContent = 'Kamera berhenti.';
                cameraResult.textContent = 'Belum ada prediksi kamera.';
                currentLabel = 'Menunggu...';
            }

            function drawOverlay(label, box) {
                const ctx = overlayCanvas.getContext('2d');
                const width = overlayCanvas.width;
                const height = overlayCanvas.height;
                ctx.clearRect(0, 0, width, height);

                const boxW = Math.min(width * 0.72, width - 20);
                const boxH = Math.min(height * 0.72, height - 20);
                const x = (width - boxW) / 2;
                const y = (height - boxH) / 2;
                const rect = box || { x, y, w: boxW, h: boxH };

                ctx.strokeStyle = '#22c55e';
                ctx.lineWidth = 4;
                ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);

                ctx.font = '18px Arial';
                const text = label || 'Mendeteksi...';
                const textWidth = ctx.measureText(text).width + 20;
                ctx.fillStyle = 'rgba(0,0,0,0.7)';
                ctx.fillRect(rect.x, Math.max(8, rect.y - 36), textWidth, 30);
                ctx.fillStyle = 'white';
                ctx.fillText(text, rect.x + 10, Math.max(28, rect.y - 14));
            }

            async function captureAndPredict() {
                if (!stream || video.readyState < 2) return;

                const width = video.videoWidth || 320;
                const height = video.videoHeight || 240;
                const boxW = Math.min(width * 0.72, 280);
                const boxH = Math.min(height * 0.72, 280);
                const x = (width - boxW) / 2;
                const y = (height - boxH) / 2;

                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = width;
                tempCanvas.height = height;
                const ctx = tempCanvas.getContext('2d');
                ctx.drawImage(video, 0, 0, width, height);

                const cropCanvas = document.createElement('canvas');
                cropCanvas.width = boxW;
                cropCanvas.height = boxH;
                const cropCtx = cropCanvas.getContext('2d');
                cropCtx.drawImage(tempCanvas, x, y, boxW, boxH, 0, 0, boxW, boxH);

                cameraStatus.textContent = 'Mendeteksi objek...';
                cropCanvas.toBlob(async (blob) => {
                    if (!blob) return;
                    const formData = new FormData();
                    formData.append('file', blob, 'camera-frame.jpg');
                    try {
                        const response = await fetch('/predict', { method: 'POST', body: formData });
                        const data = await response.json();
                        if (response.ok && data && data.prediction) {
                            currentLabel = `${data.prediction.label} (${data.prediction.confidence}%)`;
                            cameraResult.textContent = JSON.stringify(data, null, 2);
                            drawOverlay(currentLabel, { x, y, w: boxW, h: boxH });
                        } else {
                            drawOverlay('Tidak terdeteksi', { x, y, w: boxW, h: boxH });
                        }
                    } catch (error) {
                        cameraResult.textContent = 'Error: ' + error.message;
                    }
                }, 'image/jpeg', 0.9);
            }

            function startPredictionLoop() {
                if (predictionTimer) clearInterval(predictionTimer);
                predictionTimer = setInterval(captureAndPredict, 1500);
                drawOverlay('Mendeteksi...');
                captureAndPredict();
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/health")
def health_check():
    if predictor is None:
        return JSONResponse(
            status_code=503, 
            content={"status": "error", "message": "Model not loaded", "detail": load_error}
        )
    return {"status": "healthy", "model": "loaded"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        error_msg = f"Model not loaded. Error during startup: {load_error}" if load_error else "Model not loaded. Check server logs."
        raise HTTPException(status_code=503, detail=error_msg)
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Use a unique temp file to ensure thread-safety for concurrent requests
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
            image.save(temp_path)
        
        try:
            result = predictor.predict_image(temp_path)
        finally:
            # Always clean up regardless of prediction success/failure
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
        return {
            "success": True,
            "prediction": {
                "label": result['class'].replace('_', ' ').title(),
                "raw_label": result['class'],
                "confidence": round(result['confidence'] * 100, 2),
            },
            "intelligence": {
                "disposal": result.get("disposal_method", "N/A"),
                "recycling": result.get("recycling_ideas", "N/A"),
                "impact": result.get("environmental_impact", "N/A"),
                "notes": result.get("notes", "")
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = get_available_port()
    print(f"Starting API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
