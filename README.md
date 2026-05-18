# 🚗 Driver Drowsiness Detection

> Real-time driver behavior classification using **YOLO11s** — detecting 6 dangerous and safe driving states from a camera feed.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![YOLO](https://img.shields.io/badge/YOLO-11s-purple?style=flat-square)
![Ultralytics](https://img.shields.io/badge/Ultralytics-latest-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

This project trains and deploys a **YOLO11s** object detection model to classify driver behavior in real time into 6 categories. It is designed to enhance road safety by alerting when a driver shows signs of fatigue, distraction, or dangerous behavior.

---

## 🎯 Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | `DangerousDriving` | Aggressive or reckless driving behavior |
| 1 | `Distracted` | Driver attention away from the road |
| 2 | `Drinking` | Driver consuming a beverage while driving |
| 3 | `SafeDriving` | Normal, attentive driving |
| 4 | `SleepyDriving` | Driver showing signs of fatigue |
| 5 | `Yawn` | Driver yawning |

---

## 🛠️ Pipeline

```
Dataset Download → EDA & Cleaning → Visual Inspection
       ↓
Stratified 80/10/10 Re-split
       ↓
YOLO Format Conversion
       ↓
Offline Augmentation (Minority Classes)
       ↓
YOLO11s Training (50 epochs)
       ↓
Evaluation: mAP, Precision, Recall, F1
       ↓
Error Analysis (FP / FN Visualization)
```

---

## 📂 Project Structure

```
driver-drowsiness-detection/
│
├── app.py               # Inference / deployment script
├── best.pt              # Trained YOLO11s weights (best checkpoint)
└── requirements.txt     # Python dependencies
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Mohab-Refaai/driver-drowsiness-detection.git
cd driver-drowsiness-detection

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
python app.py
```

The app will load `best.pt` and run inference. You can pass a video file, webcam index, or image path depending on the script configuration.

---

## 🏋️ Training Details

| Parameter | Value |
|-----------|-------|
| Model | YOLO11s |
| Epochs | 50 |
| Image Size | 640 × 640 |
| Batch Size | 16 |
| Optimizer | AdamW (auto) |
| Early Stopping | patience = 10 |
| Augmentation | Mosaic, HFlip, HSV, Rotation |

### Dataset Split

| Split | Ratio |
|-------|-------|
| Train | 80% |
| Validation | 10% |
| Test | 10% |

Stratified splitting was applied to preserve class distribution across all splits.

### Offline Augmentation

Minority classes (`Drinking`, `SleepyDriving`, `Yawn`) were augmented with **+300 samples each** using:
- Horizontal Flip
- Random Brightness & Contrast
- Rotation (±15°)

---

## 📊 Evaluation Metrics

The model is evaluated on the test set using:

- **mAP50** — mean Average Precision @ IoU 0.50
- **mAP50-95** — mean Average Precision @ IoU 0.50–0.95
- **Precision / Recall / F1** — per class and overall

---

## 📋 Requirements

```
ultralytics
opencv-python
torch
torchvision
albumentations
scikit-learn
matplotlib
Pillow
PyYAML
```

---

## 👤 Author

**Mohab Hossam** — [@Mohab-Refaai](https://github.com/Mohab-Refaai)

---

## 📄 License

This project is licensed under the MIT License.
