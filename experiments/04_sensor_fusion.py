import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO

def run_sensor_fusion():
    """
    Experiment 04: Sensor Fusion (YOLOv8 + MiDaS).
    
    Objective:
    - Run Object Detection and Depth Estimation in parallel.
    - Extract the depth ROI (Region of Interest) for each detected object.
    - Estimate relative distance based on depth intensity.
    - Display a HUD with Object Name + Estimated Distance.
    """
    
    print("🚀 Initializing Experiment 04: Sensor Fusion...")

    # --- 1. LOAD MODELS ---
    
    # A. YOLOv8 Nano
    print("Loading YOLOv8...")
    # Update path if you moved the model to models/yolo/
    try:
        yolo_model = YOLO('models/yolo/yolov8n.pt') 
    except:
        yolo_model = YOLO('yolov8n.pt') # Fallback
        
    # B. MiDaS Small
    print("Loading MiDaS Small...")
    midas_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", midas_type)
    
    # Device selection (MPS for Mac, CUDA for Nvidia, CPU fallback)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    print(f"✅ Models loaded on {device}")

    # --- 2. MAIN LOOP ---
    cap = cv2.VideoCapture(0)
    
    # Colors for depth warning
    COLOR_SAFE = (0, 255, 0)   # Green
    COLOR_WARN = (0, 255, 255) # Yellow
    COLOR_DANGER = (0, 0, 255) # Red

    prev_frame_time = 0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret: break

            h, w, _ = frame.shape

            # --- STEP A: DEPTH ESTIMATION (MiDaS) ---
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = transform(img_rgb).to(device)
            
            prediction = midas(input_batch)
            
            # Upscale depth map to original resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            depth_map = prediction.cpu().numpy()

            # Normalize depth to 0-255 (255 = Close, 0 = Far)
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            # Avoid division by zero
            if depth_max - depth_min > 0:
                depth_norm = (depth_map - depth_min) / (depth_max - depth_min) * 255.0
            else:
                depth_norm = np.zeros_like(depth_map)
                
            depth_uint8 = depth_norm.astype(np.uint8)

            # --- STEP B: OBJECT DETECTION (YOLO) ---
            results = yolo_model(frame, verbose=False, conf=0.5)
            
            # --- STEP C: FUSION ---
            # Instead of plotting YOLO directly, we draw manually to add depth info
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # 1. Get Coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label_name = yolo_model.names[cls]

                    # 2. Extract Depth ROI (Region of Interest)
                    # We look at the center of the bounding box (more reliable)
                    # or the median of the whole box. Let's use Median of center 50%
                    
                    center_x1 = int(x1 + (x2 - x1) * 0.25)
                    center_x2 = int(x2 - (x2 - x1) * 0.25)
                    center_y1 = int(y1 + (y2 - y1) * 0.25)
                    center_y2 = int(y2 - (y2 - y1) * 0.25)
                    
                    # Safety check for image bounds
                    center_x1 = max(0, center_x1)
                    center_y1 = max(0, center_y1)
                    
                    # Crop depth map
                    depth_roi = depth_uint8[center_y1:center_y2, center_x1:center_x2]
                    
                    if depth_roi.size > 0:
                        # Calculate Median Intensity (Robust to noise)
                        median_depth = np.median(depth_roi)
                    else:
                        median_depth = 0

                    # 3. Interpret Distance (Heuristic)
                    # Higher value = Closer
                    if median_depth > 180:
                        dist_label = "VERY CLOSE (<1m)"
                        box_color = COLOR_DANGER
                    elif median_depth > 100:
                        dist_label = "NEAR (1-3m)"
                        box_color = COLOR_WARN
                    else:
                        dist_label = "FAR (>3m)"
                        box_color = COLOR_SAFE

                    # 4. Draw HUD
                    # Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    
                    # Label Background
                    label_text = f"{label_name} | {dist_label}"
                    (w_text, h_text), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + w_text, y1), box_color, -1)
                    
                    # Text
                    cv2.putText(frame, label_text, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # --- STEP D: DISPLAY ---
            # Calculate FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            
            # Small Depth Map preview in the corner
            depth_colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
            depth_small = cv2.resize(depth_colormap, (160, 120))
            frame[0:120, 0:160] = depth_small # Overlay

            cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 120, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow('VisionAssistant - Exp 04 (Fusion)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Experiment finished.")

if __name__ == "__main__":
    run_sensor_fusion()