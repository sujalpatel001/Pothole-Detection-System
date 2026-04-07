# Pothole Detection System

A simple AI-driven system for detecting potholes in road images using a YOLOv8-based model.

## 📁 Files in This Project

- **pothole_detection.py**: The main script that processes images and detects potholes.
- **download_model.py**: A utility script to download the pre-trained model weights (`best.pt`) from HuggingFace.
- **best.pt**: The YOLOv8 model file (weights).
- **images/**: Put your input images (road photos) in this folder.
- **output_images/**: The annotated results will be saved here.

## 🚀 How to Run

1. **Download the Model** (if `best.pt` is missing):
   ```bash
   python download_model.py
   ```

2. **Run Detection**:
   Place your images in the `images/` directory, then run:
   ```bash
   python pothole_detection.py
   ```

The system will process each image and save the result with bounding boxes and segmentations in the `output_images/` folder.
