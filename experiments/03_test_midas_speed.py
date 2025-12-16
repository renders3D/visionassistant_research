import cv2
import torch
import time
import numpy as np

def run_midas_test():
    """
    Experiment 03: High-Speed Depth Estimation with MiDaS Small.
    
    Objective:
    - Replace the heavy Transformer model with MiDaS v2.1 Small (CNN).
    - Achieve >10 FPS on CPU for real-time navigation compatibility.
    """
    
    print("🚀 Initializing Experiment 03: MiDaS Speed Test...")

    # 1. Load Model from Torch Hub (Intel Intelligent Systems Lab)
    # "MiDaS_small" is optimized for mobile/edge inference.
    print("🔄 Loading MiDaS Small model...")
    try:
        model_type = "MiDaS_small"
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        
        # Move to GPU if available (MPS for Mac M1, CUDA for Nvidia, else CPU)
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        # Fallback to CPU if MPS gives trouble with some ops
        # device = torch.device("cpu") 
        
        midas.to(device)
        midas.eval() # Set to evaluation mode (freezes weights)
        
        # Load the specific transform for this model
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.small_transform

        print(f"✅ Model loaded on: {device}")
    except Exception as e:
        print(f"❌ Error loading MiDaS: {e}")
        return

    # 2. Webcam
    cap = cv2.VideoCapture(0)
    print("🎥 Comparison started. Press 'Q' to exit.")

    prev_frame_time = 0
    new_frame_time = 0

    with torch.no_grad(): # Disable gradient calculation for speed
        while True:
            ret, frame = cap.read()
            if not ret: break

            # 3. Preprocess
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply MiDaS transforms (Resize, Normalize)
            input_batch = transform(img).to(device)

            # 4. Inference
            prediction = midas(input_batch)

            # 5. Resize to original resolution
            # MiDaS outputs relative inverse depth.
            # We use bicubic interpolation to scale it back up.
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            # 6. Post-process for Visualization
            depth_map = prediction.cpu().numpy()
            
            # Normalize to 0-255
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            depth_norm = (depth_map - depth_min) / (depth_max - depth_min) * 255.0
            depth_uint8 = depth_norm.astype(np.uint8)

            # Color Map
            depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)

            # FPS Calculation
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            # 7. Display
            # Stack images
            combined = np.hstack((frame, depth_colored))
            
            # HUD
            cv2.putText(combined, f"FPS: {int(fps)}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, f"Model: {model_type}", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('VisionAssistant - Exp 03 (Fast Depth)', combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Experiment finished.")

if __name__ == "__main__":
    run_midas_test()