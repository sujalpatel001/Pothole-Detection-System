# 🕳️ Pothole Detection System (YOLOv8 Medium)

An automated, deep-learning based Python application designed to detect and bounding-box potholes in unlabelled road images. 

This project uses a pretrained **YOLOv8 Medium** architecture (`keremberke/yolov8m-pothole-segmentation`) containing ~25 million parameters, enabling high-accuracy context-aware detections. It actively distinguishes between true road hazards (potholes) and visual distractors like shadows, manhole covers, and car tires.

---

## 🚀 Key Features

* **Multi-Scale Inference Pattern**: To maximize "recall" (so no pothole hides from the AI), it natively scans your images at 3 distinct internal resolutions (`640px`, `960px`, `1280px`). This guarantees that both large close-up craters and tiny background potholes are found.
* **Non-Maximum Suppression (NMS)**: Because scanning the image 3 times creates multiple overlapping bounding boxes, it uses an advanced computer vision deduplication algorithm (`cv2.dnn.NMSBoxes`) to collapse the extra boxes into perfectly aligned detection zones.
* **Dynamic Visualization**: The script draws vivid red bounding boxes over the damaged road areas, stamps the exact AI confidence `%` dynamically above the pothole, and sums the total counts in a heads-up-display badge.
* **TTA Ready / False Positive Resistant**: The heavy internal weights of the `medium` model actively reject features that visually trick lightweight models.

---

## 🛠️ Requirements
You will need an active Python 3.10+ environment installed.
To install the dependencies, simply run the following in your terminal:
```bash
pip install torch torchvision opencv-python numpy Pillow
pip install ultralyticsplus==0.0.23 ultralytics==8.0.21
```

---

## 📁 System Architecture
```text
P1/
├── images/                   ← Drop your road images here (.jpg, .png, etc.)
├── output_images/            ← Annotated results appear here automatically
├── download_model.py         ← Script to fetch the Heavy YOLOv8m weights securely
├── pothole_detection.py      ← The main multi-scale detection engine
└── best_m.pt                 ← (Auto-downloaded) The 52MB pretrained brain
```

---

## 🤔 How to Run

1. **Prepare your files**
   Place any images you want to analyze freely into the `images/` directory.

2. **Download AI Weights (First Run Only)**
   The AI model (`best_m.pt`) is quite large. Run the auto-downloader to fetch it from Hugging Face:
   ```bash
   python download_model.py
   ```

3. **Launch the Engine**
   Run the main script. The system will process everything in the `images` folder seamlessly:
   ```bash
   python pothole_detection.py
   ```

4. **View the Results**
   Head to `output_images/` when it prints `Done!` to see your images professionally annotated.

---

### Adjusting Hyperparameters
If you wish to make the AI stricter or more relaxed, you can edit the top section of `pothole_detection.py`:
- `CONF_THRESH` (Default `0.01`): The confidence floor. Lower values capture very subtle potholes but run the risk of false positives.
- `IOU_THRESH` (Default `0.50`): Controls how aggressively overlapping boxes are merged together.

*Powered by PyTorch & Ultralytics*
