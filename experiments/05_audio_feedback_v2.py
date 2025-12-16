import cv2
import torch
import numpy as np
import time
import threading
import sys
import subprocess
from ultralytics import YOLO

# --- AUDIO ENGINE ROBUSTO (MAC M1 FRIENDLY) ---
class VoiceAssistant:
    def __init__(self):
        self.last_spoken_time = 0
        self.normal_cooldown = 5.0      # Hablar cada 5 segs si es normal
        self.emergency_cooldown = 3.0   # Hablar cada 3 segs si es peligro (NO ANTES)
        
        # Detectar si estamos en Mac
        self.is_mac = sys.platform == 'darwin'
        
        if not self.is_mac:
            # Fallback para Windows/Linux
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 160)

    def _speak_mac(self, text):
        """Usa el comando nativo de Mac 'say'. Es no bloqueante y muy estable."""
        try:
            # Popen lanza el proceso en fondo y no bloquea el video
            subprocess.Popen(['say', text])
        except Exception as e:
            print(f"Audio error: {e}")

    def _speak_generic(self, text):
        """Para Windows/Linux usando pyttsx3"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except:
            pass

    def announce(self, text, is_emergency=False):
        """Gestión inteligente de tiempos para no saturar al usuario"""
        current_time = time.time()
        
        # Elegir qué cooldown aplicar
        limit = self.emergency_cooldown if is_emergency else self.normal_cooldown
        
        # Solo hablamos si ha pasado el tiempo suficiente
        if current_time - self.last_spoken_time > limit:
            print(f"🗣️ Assistant: '{text}'")
            
            if self.is_mac:
                self._speak_mac(text)
            else:
                # En Windows/Linux necesitamos hilos para no congelar el video
                t = threading.Thread(target=self._speak_generic, args=(text,))
                t.start()
            
            self.last_spoken_time = current_time

def run_audio_experiment_v2():
    print("🚀 Initializing Experiment 05 v2: Stable Audio...")

    # 1. Init Voice
    voice = VoiceAssistant()
    voice.announce("System initializing...", is_emergency=False)

    # 2. Load Models
    try:
        yolo_model = YOLO('models/yolo/yolov8n.pt') 
    except:
        yolo_model = YOLO('yolov8n.pt')

    # Load MiDaS
    midas_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", midas_type)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    print(f"✅ Models loaded on {device}")
    voice.announce("Ready to navigate.", is_emergency=False)

    # 3. Main Loop
    cap = cv2.VideoCapture(0)
    
    COLOR_SAFE = (0, 255, 0)
    COLOR_WARN = (0, 255, 255)
    COLOR_DANGER = (0, 0, 255)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- DEPTH ESTIMATION ---
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img_rgb).to(device)
        
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        depth_map = prediction.cpu().numpy()
        
        # Normalize
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max - depth_min > 0:
            depth_uint8 = ((depth_map - depth_min) / (depth_max - depth_min) * 255.0).astype(np.uint8)
        else:
            depth_uint8 = np.zeros_like(depth_map, dtype=np.uint8)

        # --- OBJECT DETECTION ---
        results = yolo_model(frame, verbose=False, conf=0.5)
        
        # --- LOGIC ---
        closest_intensity = -1
        priority_label = ""
        priority_zone = "" 

        frame_h, frame_w, _ = frame.shape
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label_name = yolo_model.names[cls]

                # Depth Analysis
                cx1, cx2 = int(x1 + (x2-x1)*0.25), int(x2 - (x2-x1)*0.25)
                cy1, cy2 = int(y1 + (y2-y1)*0.25), int(y2 - (y2-y1)*0.25)
                
                depth_roi = depth_uint8[cy1:cy2, cx1:cx2]
                if depth_roi.size > 0:
                    median_depth = np.median(depth_roi)
                else:
                    median_depth = 0

                # Zone Analysis
                cx = (x1 + x2) // 2
                if cx < frame_w * 0.33: zone = "on your left"
                elif cx > frame_w * 0.66: zone = "on your right"
                else: zone = "ahead"

                # Priority Logic: 
                # We care about the CLOSEST object.
                if median_depth > closest_intensity:
                    closest_intensity = median_depth
                    priority_label = label_name
                    priority_zone = zone

                # Visuals
                color = COLOR_SAFE
                if median_depth > 180: color = COLOR_DANGER
                elif median_depth > 100: color = COLOR_WARN
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label_name}", (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- AUDIO TRIGGER ---
        # Solo hablamos si hay algo relevante (> 100 intensidad)
        if closest_intensity > 100:
            msg = f"{priority_label} {priority_zone}"
            
            # Si está MUY cerca (> 180), es emergencia
            if closest_intensity > 180:
                warning = f"Stop! {priority_label} very close"
                # Esto usará el cooldown de emergencia (3s)
                voice.announce(warning, is_emergency=True)
            else:
                # Esto usará el cooldown normal (5s)
                voice.announce(msg, is_emergency=False)

        # --- DISPLAY ---
        # Mini mapa de profundidad
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
        depth_mini = cv2.resize(depth_color, (160, 120))
        frame[0:120, 0:160] = depth_mini

        cv2.imshow('VisionAssistant v2 (Stable)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Experiment finished.")

if __name__ == "__main__":
    run_audio_experiment_v2()