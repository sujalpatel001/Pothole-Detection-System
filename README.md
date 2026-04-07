# 🛣️ Road Sentinel: AI Pothole Detection System

## 📌 Overview
**Road Sentinel** is a high-performance, deep learning-based system designed to automatically detect and segment potholes from road imagery. Utilizing a custom-trained **YOLOv8** model, this project aims to assist in road maintenance and safety by providing precise location and segmentation data for road defects.

## ✨ Key Features
- **🎯 Precise Segmentation**: Beyond simple bounding boxes, it generates masks to pinpoint the exact area of the pothole.
- **🚀 Rapid Inference**: Optimized for YOLOv8n (nano), ensuring fast processing of large batches of images.
- **📊 Detailed Annotations**: Automatically overlays confidence scores and summary badges on processed images.
- **🛠️ Automated Setup**: Built-in dependency guards and model downloaders for a seamless "plug-and-play" experience.

## 📈 Model Performance
- **Model Architecture**: YOLOv8n (Segmentation)
- **Mean Average Precision (mAP@0.5)**: 0.995
- **Dataset Source**: [keremberke/pothole-segmentation](https://huggingface.co/keremberke/yolov8n-pothole-segmentation)

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8+
- [Git](https://git-scm.com/)

### 2. Installation
Clone the repository and install the dependencies:
```bash
# Clone the repository
git clone <repository-url>
cd "FINAL - POTHOLE DETECTION SYS"

# Install requirements
pip install -r requirements.txt
```

### 3. Download the Model
Run the download script to fetch the `best.pt` model weights from HuggingFace:
```bash
python download_model.py
```

### 4. Run Detection
Place your road images (JPG, PNG, WebP) in the `images/` directory, then execute the main script:
```bash
python pothole_detection.py
```
The annotated results will be saved in the `output_images/` directory.

## 📂 Project Structure
```text
.
├── docs/                   # Documentation assets (hero image, etc.)
├── images/                 # 📂 Put your input images here
├── output_images/          # 📂 Processed images will appear here
├── best.pt                 # 🧠 YOLOv8n Model Weights
├── download_model.py       # 📥 Script to fetch model weights
├── pothole_detection.py    # ⚙️ Main detection script
├── requirements.txt        # 📦 Project dependencies
└── README.md               # 📖 This file
```

## 🛠️ Visual Style
The system uses high-visibility red overlays for detected potholes:
- **Bounding Boxes**: Clear red rectangles for location.
- **Segmentation Masks**: Subtle red fill for surface area coverage.
- **Summary Badge**: Top-left corner counter for total detections in the frame.

## 📜 Credits
Developed using the [Ultralytics](https://github.com/ultralytics/ultralytics) framework and the [keremberke/yolov8n-pothole-segmentation](https://huggingface.co/keremberke/yolov8n-pothole-segmentation) model available on HuggingFace.

---
*Ensuring safer roads through Artificial Intelligence.*
