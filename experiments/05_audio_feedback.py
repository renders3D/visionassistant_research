import cv2
import torch
import numpy as np
import time
import pyttsx3
import threading
from ultralytics import YOLO

# --- AUDIO ENGINE SETUP ---
class VoiceAssistant:
    def __init__(self):
        self.engine = pyttsx3.init()
        # Adjust rate (speed) and volume
        self.engine.setProperty('rate', 160) 
        self.engine.setProperty('volume', 1.0)
        self.last_spoken_time = 0
        self.cooldown = 4.0 # Seconds between announcements for the same object

    def speak_thread(self, text):
        """Worker function to run in a separate thread"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except RuntimeError:
            # Handle potential loop overlap errors
            pass

    def announce(self, text, force=False):
        """Non-blocking speech announcement"""
        current_time = time.time()
        
        # Logic: Only speak if enough time has passed OR if it's an emergency (force)
        if force or (current_time - self.last_spoken_time > self.cooldown):
            print(f"🗣️ Assistant says: '{text}'")
            # Create a thread so video doesn't freeze
            t = threading.Thread(target=self.speak_thread, args=(text,))
            t.start()
            self.last_spoken_time = current_time

def run_audio_experiment():
    print("🚀 Initializing Experiment 05: Full Audio Integration...")

    # 1. Init Voice
    voice = VoiceAssistant()
    voice.announce("System starting. Loading vision models.", force=True)

    # 2. Load Models
    try:
        yolo_model = YOLO('models/yolo/yolov8n.pt') 
    except:
        yolo_model = YOLO('yolov8n.pt')

    midas_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", midas_type)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    print(f"✅ Models loaded on {device}")
    voice.announce("Vision ready.", force=True)

    # 3. Main Loop
    cap = cv2.VideoCapture(0)
    
    # Colors
    COLOR_SAFE = (0, 255, 0)
    COLOR_WARN = (0, 255, 255)
    COLOR_DANGER = (0, 0, 255)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- DEPTH ---
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
        
        # Normalize 0-255
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            depth_uint8 = ((depth_map - depth_min) / (depth_max - depth_min) * 255.0).astype(np.uint8)
        else:
            depth_uint8 = np.zeros_like(depth_map, dtype=np.uint8)

        # --- DETECTION ---
        results = yolo_model(frame, verbose=False, conf=0.5)
        
        # --- SPATIAL LOGIC & AUDIO TRIGGER ---
        closest_dist = -1
        closest_label = ""
        closest_zone = "" # "Center", "Left", "Right"

        frame_h, frame_w, _ = frame.shape
        center_x_img = frame_w // 2

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label_name = yolo_model.names[cls]

                # Extract Depth
                cx1 = int(x1 + (x2 - x1) * 0.25)
                cx2 = int(x2 - (x2 - x1) * 0.25)
                cy1 = int(y1 + (y2 - y1) * 0.25)
                cy2 = int(y2 - (y2 - y1) * 0.25)
                
                depth_roi = depth_uint8[cy1:cy2, cx1:cx2]
                if depth_roi.size > 0:
                    median_depth = np.median(depth_roi)
                else:
                    median_depth = 0

                # Determine Zone (Left/Center/Right)
                obj_center_x = (x1 + x2) // 2
                if obj_center_x < frame_w * 0.33:
                    zone = "on your left"
                elif obj_center_x > frame_w * 0.66:
                    zone = "on your right"
                else:
                    zone = "in front of you"

                # Logic: Find the most dangerous object (Highest Median Depth = Closest)
                if median_depth > closest_dist:
                    closest_dist = median_depth
                    closest_label = label_name
                    closest_zone = zone

                # Visualization
                color = COLOR_SAFE
                dist_msg = "Far"
                if median_depth > 180: 
                    color = COLOR_DANGER
                    dist_msg = "VERY CLOSE"
                elif median_depth > 100: 
                    color = COLOR_WARN
                    dist_msg = "Near"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label_name} | {dist_msg}", (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- DECISION MAKING ---
        # Only speak if there is a relevant object closer than threshold (e.g. 100 intensity)
        if closest_dist > 100:
            # Construct natural language sentence
            # e.g., "Person in front of you"
            message = f"{closest_label} {closest_zone}"
            
            # If VERY close, add warning
            if closest_dist > 180:
                message = f"Stop! {closest_label} very close {closest_zone}"
                voice.announce(message, force=True) # Force interrupt
            else:
                voice.announce(message) # Normal cooldown

        # Display
        cv2.imshow('VisionAssistant - Exp 05 (Audio)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Experiment finished.")

if __name__ == "__main__":
    run_audio_experiment()