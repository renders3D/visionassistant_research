# 👁️ VisionAssistant: Acoustic Navigation for the Visually Impaired

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Computer Vision](https://img.shields.io/badge/Computer_Vision-YOLO_%2B_Depth-green)
![Status](https://img.shields.io/badge/Status-Research_Phase-orange)

**VisionAssistant** is an applied Computer Vision research project aimed at creating a real-time navigational aid for visually impaired individuals. The system leverages monocular cameras (smartphones) to understand 3D space, detect obstacles, and provide spatial audio feedback.

## 🎯 Research Objectives

1.  **Object Detection:** Implement SOTA models (YOLOv8/v10) to identify key navigational elements (doors, stairs, people, obstacles).
2.  **Monocular Depth Estimation:** Utilize Transformers (Depth Anything / MiDaS) to compute a relative depth map from a single RGB image.
3.  **Spatial Logic:** Develop algorithms to combine 2D bounding boxes with depth maps to estimate real-world distance (metric depth).
4.  **Acoustic Feedback:** Translate spatial data into natural language descriptions (e.g., *"Door on your left, 2 meters away"*).

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Detection:** Ultralytics YOLOv8
* **Depth Estimation:** HuggingFace Transformers (Depth Anything V2)
* **Image Processing:** OpenCV, NumPy
* **Visualization:** Matplotlib, Open3D

## 📂 Project Structure

```text
VisionAssistant_Research/
├── data/           # Datasets and sample videos
├── models/         # Model weights (YOLO .pt, Depth encoders)
├── notebooks/      # Jupyter notebooks for experimentation
├── src/            # Core source code
│   ├── detection/  # Object detection logic
│   ├── depth/      # Depth estimation logic
│   └── utils/      # Helper functions
└── experiments/    # Scripts for specific R&D tests
```

## 🚀 Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the first experiment (Webcam Test):
    ```bash
    python experiments/01_test_yolo_setup.py
    ```

---
*Research Lead: Carlos Luis Noriega*
