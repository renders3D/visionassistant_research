import os

def create_structure():
    # Define the project structure
    structure = {
        "data": ["raw", "processed", "samples"],
        "models": ["yolo", "depth"],
        "notebooks": [],
        "src": ["detection", "depth", "utils"],
        "experiments": [],
        "docs": []
    }

    # Create directories
    print("🚀 Initializing 'VisionAssistant_Research' environment...")
    
    for main_folder, subfolders in structure.items():
        if not os.path.exists(main_folder):
            os.makedirs(main_folder)
            print(f"   [+] Created directory: {main_folder}/")
        
        for sub in subfolders:
            path = os.path.join(main_folder, sub)
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"   [+] Created directory: {path}/")
                
                # Create __init__.py in src submodules to make them packages
                if main_folder == "src":
                    with open(os.path.join(path, "__init__.py"), 'w') as f:
                        f.write(f"# Module: {sub}\n")

    # Create root __init__.py for src
    if not os.path.exists("src/__init__.py"):
        with open("src/__init__.py", 'w') as f:
            f.write("# Source code for VisionAssistant\n")

    # Create .gitignore
    create_gitignore()
    
    # Create README.md
    create_readme()

    print("\n✅ Project setup complete. Ready for Research!")

def create_gitignore():
    content = """# --- Python ---
__pycache__/
*.py[cod]
*$py.class
venv/
.venv/
env/
.env

# --- Data & Models (Do not sync large files) ---
data/raw/*
data/processed/*
models/**/*.pt
models/**/*.onnx
models/**/*.tflite
models/**/*.h5

# Keep gitkeep to track folder structure
!data/raw/.gitkeep
!models/.gitkeep

# --- IDEs & OS ---
.vscode/
.idea/
.DS_Store
Thumbs.db

# --- Jupyter ---
.ipynb_checkpoints
"""
    with open(".gitignore", "w") as f:
        f.write(content)
    print("   [+] Created .gitignore")

def create_readme():
    content = """# 👁️ VisionAssistant: Acoustic Navigation for the Visually Impaired

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
"""
    with open("README.md", "w") as f:
        f.write(content)
    print("   [+] Created README.md (English)")

if __name__ == "__main__":
    create_structure()