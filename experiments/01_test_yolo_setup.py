import cv2
from ultralytics import YOLO
import time

def run_yolo_test():
    """
    Experiment 01: Basic YOLOv8 Setup Validation.
    
    Objective:
    - Verify webcam access.
    - Download and load the YOLOv8 Nano model (lightweight).
    - Perform real-time inference on the video stream.
    """
    
    print("🚀 Initializing Experiment 01: YOLOv8 Setup...")

    # 1. Load the Model
    # We use 'yolov8n.pt' (Nano) for the fastest CPU inference.
    # It will download automatically on the first run.
    print("🔄 Loading YOLOv8 Nano model...")
    try:
        model = YOLO('models/yolo/yolov8n.pt')
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Initialize Webcam
    # Index 0 is usually the default built-in webcam.
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("🎥 Webcam started. Press 'Q' to exit.")

    # FPS Calculation variables
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame.")
            break

        # 3. Inference
        # 'stream=True' is more memory efficient for video generators, 
        # but direct call is fine for simple loops.
        # conf=0.5 filters out low-confidence predictions.
        results = model(frame, conf=0.5, verbose=False)

        # 4. Visualization
        # The 'plot()' method draws the bounding boxes and labels automatically.
        annotated_frame = results[0].plot()

        # Calculate FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        
        # Display FPS on screen
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('VisionAssistant - Experiment 01 (YOLOv8n)', annotated_frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Experiment finished.")

if __name__ == "__main__":
    run_yolo_test()