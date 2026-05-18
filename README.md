# 🚗 Driver Drowsiness Detection

> Real-time driver behavior classification using **YOLO11s** — detecting 6 dangerous and safe driving states from a camera feed.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![YOLO](https://img.shields.io/badge/YOLO-11s-purple?style=flat-square)
![mAP50](https://img.shields.io/badge/mAP50-98.9%25-brightgreen?style=flat-square)
![F1](https://img.shields.io/badge/F1-97.7%25-brightgreen?style=flat-square)
![GPU](https://img.shields.io/badge/GPU-Tesla%20T4-orange?style=flat-square)

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

## 📊 Results

### Overall Metrics (Test Set)

| Metric | Value |
|--------|-------|
| **mAP50** | **98.9%** |
| **mAP50-95** | **87.3%** |
| **Precision** | **97.5%** |
| **Recall** | **97.9%** |
| **F1 Score** | **97.7%** |

### Per-Class Results (Test Set)

| Class | Precision | Recall | F1 | mAP50 | mAP50-95 |
|-------|-----------|--------|----|-------|----------|
| DangerousDriving | 99.7% | 99.6% | 99.7% | 99.5% | 89.8% |
| Distracted | 97.9% | 95.2% | 96.5% | 99.0% | 85.8% |
| Drinking | 95.3% | 100.0% | 97.6% | 99.2% | 80.9% |
| SafeDriving | 98.9% | 99.7% | 99.3% | 99.4% | 88.4% |
| SleepyDriving | 95.0% | 94.9% | 94.9% | 98.0% | 90.4% |
| Yawn | 98.1% | 98.2% | 98.1% | 98.2% | 88.8% |

> **Inference Speed:** ~5.6ms per image on Tesla T4 GPU

---

## 🛠️ Pipeline

```
Dataset Download (14,855 images)
       ↓
EDA & Data Cleaning (removed 4 corrupted samples)
       ↓
Stratified 80 / 10 / 10 Re-split
       ↓
YOLO Format Conversion
       ↓
Offline Augmentation (+300 samples for minority classes)
       ↓
YOLO11s Training (50 epochs, Tesla T4)
       ↓
Evaluation: mAP50 = 98.9%, F1 = 97.7%
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

---

## 🏋️ Training Details

| Parameter | Value |
|-----------|-------|
| Model | YOLO11s |
| Pretrained | ✅ (yolo11s.pt) |
| Epochs | 50 |
| Image Size | 640 × 640 |
| Batch Size | 16 |
| Optimizer | AdamW (auto) |
| Early Stopping | patience = 10 |
| GPU | Tesla T4 |
| Framework | Ultralytics 8.4.47 |
| Parameters | 9,430,114 |

### Augmentation

**Online (YOLO):** HSV shift, horizontal flip, mosaic, rotation ±10°

**Offline (minority classes only):** +300 samples each for `Drinking`, `SleepyDriving`, `Yawn` using horizontal flip, brightness/contrast, and rotation ±15°

### Dataset Split

| Split | Images | After Augmentation |
|-------|--------|--------------------|
| Train | 11,884 | 12,784 |
| Validation | 1,485 | 1,485 |
| Test | 1,486 | 1,486 |
| **Total** | **14,855** | **15,755** |

### Class Distribution (Train)

| Class | Original | After Augmentation |
|-------|----------|--------------------|
| DangerousDriving | 3,714 | 3,714 |
| Distracted | 1,664 | 1,664 |
| Drinking | 342 | 642 |
| SafeDriving | 4,944 | 4,944 |
| SleepyDriving | 783 | 1,083 |
| Yawn | 437 | 737 |

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

## 📄 License

This project is licensed under the MIT License.
