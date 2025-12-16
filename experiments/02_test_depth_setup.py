import cv2
import torch
import numpy as np
from transformers import pipeline
from PIL import Image
import time

def run_depth_test():
    """
    Experiment 02: Monocular Depth Estimation Setup.
    
    Objective:
    - Load 'Depth Anything' (Small version) via Hugging Face Transformers.
    - Process webcam feed to generate a Depth Map.
    - Visualize the 3D perception using a heat colormap.
    
    Note: Running Transformers on CPU might be slower than YOLO.
    """
    
    print("🚀 Initializing Experiment 02: Depth Estimation...")

    # 1. Load the Depth Model
    # We use 'LiheYoung/depth-anything-small-hf' for a balance of speed and accuracy.
    # The pipeline handles preprocessing (resize/normalize) and postprocessing.
    print("🔄 Loading Depth Estimation model (this may take a minute)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   [+] Computing Device: {device.upper()}")
    
    try:
        # We specify the model explicitly to ensure we get the lightweight version
        depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device=0 if device=="cuda" else -1)
        print("✅ Depth Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Initialize Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("🎥 Depth perception started. Press 'Q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 3. Preprocessing
        # Transformers pipeline expects a PIL Image (RGB), but OpenCV gives numpy (BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # 4. Inference
        # Returns a dict with 'depth' (PIL Image) or 'predicted_depth' (tensor)
        start_time = time.time()
        result = depth_estimator(pil_image)
        end_time = time.time()
        
        # 5. Post-processing for Visualization
        # The result['depth'] is a PIL image of the depth map.
        depth_map = result["depth"]
        
        # Convert PIL -> Numpy
        depth_np = np.array(depth_map)
        
        # Normalize to 0-255 for visualization (OpenCV needs uint8)
        # Formula: (value - min) / (max - min) * 255
        depth_min = depth_np.min()
        depth_max = depth_np.max()
        depth_normalized = (depth_np - depth_min) / (depth_max - depth_min) * 255.0
        depth_uint8 = depth_normalized.astype(np.uint8)
        
        # Apply a Color Map (INFERNO is great for depth: Black=Far, Yellow=Close)
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        
        # Resize depth map to match original frame (sometimes models output smaller maps)
        depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))

        # Calculate FPS
        fps = 1 / (end_time - start_time)

        # 6. Display (Side by Side)
        # Stack images horizontally: [Original Video | Depth Heatmap]
        combined_view = np.hstack((frame, depth_colored))
        
        cv2.putText(combined_view, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('VisionAssistant - Exp 02 (Depth Perception)', combined_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Experiment finished.")

if __name__ == "__main__":
    run_depth_test()