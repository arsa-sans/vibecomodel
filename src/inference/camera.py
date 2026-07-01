import cv2
import torch
import time
import numpy as np
from src.inference.predict import Predictor

class CameraAssistant:
    """Real-time waste assistant using camera feed."""
    def __init__(self, predictor, camera_id=0):
        self.predictor = predictor
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        
    def run(self):
        """Run real-time inference loop."""
        print("Starting waste assistant camera... Press 'q' to quit.")
        
        last_prediction = None
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
                
            # Perform inference every 15 frames for performance
            if frame_count % 15 == 0 or last_prediction is None:
                from PIL import Image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                
                # Temporary path for internal processing (or we could update Predictor to take PIL)
                # For now, we'll quickly update Predictor's predict_image or just use it here
                image_np = np.array(pil_img)
                transformed = self.predictor.transform(image=image_np)["image"]
                input_tensor = transformed.unsqueeze(0).to(self.predictor.model_wrapper.device)
                
                predicted_idx, confidence = self.predictor.model_wrapper.predict(input_tensor)
                class_name = self.predictor.class_names[predicted_idx.item()]
                
                last_prediction = {
                    "class": class_name,
                    "confidence": confidence.item()
                }
                if class_name in self.predictor.knowledge:
                    last_prediction.update(self.predictor.knowledge[class_name])
            
            # Overlay info
            if last_prediction:
                y_offset = 40
                cls_name = last_prediction['class'].replace('_', ' ').title()
                conf = last_prediction['confidence'] * 100
                
                cv2.putText(frame, f"{cls_name} ({conf:.1f}%)", (20, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                if 'disposal_method' in last_prediction:
                    y_offset += 30
                    cv2.putText(frame, f"How to: {last_prediction['disposal_method'][:50]}...", 
                                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Waste Assistant - AI Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()
