# 👕 Multi-Object Apparel Detection and Instance Segmentation

## 📌 Project Overview

This project implements a complete visual recognition pipeline for **multi-object clothing analysis** using the DeepFashion2 dataset.

The system performs three major tasks:

1. **Multi-label Classification** – Predict all clothing categories present in an image.
2. **Object Detection** – Predict bounding boxes and clothing category labels.
3. **Instance Segmentation** – Generate pixel-level masks for each clothing item.

This project was developed as part of the *Visual Recognition Mini Project*.

---

## 🗂 Dataset

We use the **DeepFashion2 dataset**, which provides:

- Person-centric clothing images
- JSON annotations per image
- Bounding boxes
- Category labels (13 total categories)
- Polygon segmentation masks
- Landmark annotations

### 🔹 Preprocessing

- Selected **Top 5 most frequent categories**
- Parsed JSON annotations
- Converted polygon masks to binary masks
- Generated:
  - Multi-label targets for classification
  - Bounding boxes + masks for detection/segmentation
- Applied train/validation/test splits
- Addressed class imbalance using augmentation

---

## 🧠 Architectures Implemented

### 1️⃣ Multi-Label Classification

- **Model:** ResNet-50 (Pretrained & From Scratch)
- **Activation:** Sigmoid
- **Loss:** Binary Cross Entropy (BCEWithLogitsLoss)

---

### 2️⃣ Detection + Instance Segmentation

We trained and evaluated:

#### ✅ Mask R-CNN
- Backbone: ResNet-50 + FPN
- Region Proposal Network (RPN)
- ROI Classification Head
- Bounding Box Regression Head
- Mask Head

#### ✅ YOLO (Detection + Segmentation)
- Multi-scale detection head
- Prototype mask segmentation head

---

### 3️⃣ Semantic Segmentation

#### ✅ U-Net
- Encoder-Decoder architecture
- Skip connections
- Post-processing:
  - Connected component analysis
  - Bounding box extraction
  - Majority voting for category assignment

---

## ⚙️ Training Strategy

Two training strategies were used:

1. **Training from scratch**
2. **Transfer learning**
   - Pretrained weights
   - Freeze early layers
   - Fine-tune classification/detection heads

All experiments were conducted on:
- Google Colab / Kaggle
- Models under compute constraints (< 7B parameters)

---

## 📊 Evaluation Metrics

### 🔹 Classification
- Per-class Precision
- Per-class Recall
- Per-class F1-score
- Macro F1
- Micro F1
- ROC Curves
- AUC

### 🔹 Segmentation
- Mean Intersection over Union (mIoU)
- Dice Coefficient

### 🔹 Detection
- COCO-style mAP@[0.5:0.95]
- Per-class AP
- F1-score
- ROC and AUC

---

## 📁 Repository Structure

```
├── data_loader.py
├── train_classifier.py
├── train_maskrcnn.py
├── train_yolo.py
├── train_unet.py
├── evaluation/
│   ├── classification_metrics.py
│   ├── detection_metrics.py
│   ├── segmentation_metrics.py
├── models/
│   ├── resnet_classifier.py
│   ├── maskrcnn_model.py
│   ├── unet_model.py
├── notebooks/
├── results/
├── README.md
```

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install torch torchvision
pip install pycocotools
pip install scikit-learn
pip install opencv-python
```

---

### 2️⃣ Train Multi-label Classifier

```bash
python train_classifier.py
```

---

### 3️⃣ Train Mask R-CNN

```bash
python train_maskrcnn.py
```

---

### 4️⃣ Train U-Net

```bash
python train_unet.py
```

---

## 📈 Results Summary

| Model        | Task            | Metric Highlights |
|-------------|----------------|-------------------|
| ResNet-50   | Classification | Macro F1, AUC     |
| Mask R-CNN  | Detection + Seg | mAP, mIoU         |
| YOLO        | Detection + Seg | Fast inference    |
| U-Net       | Segmentation   | Dice, mIoU        |

*(Full results available in report PDF.)*

---

## 🤖 Model Release

The best-performing models are uploaded to Hugging Face:

- Classification Model
- Detection + Segmentation Model

Inference scripts are provided for reproducibility.

---

## 📌 Key Observations

- Transfer learning significantly improved convergence speed.
- Mask R-CNN achieved higher segmentation accuracy than YOLO.
- U-Net performed well for large clothing regions but struggled with small objects.
- Class imbalance affected rare clothing categories.

---

## 🧾 Final Report

The final report includes:
- Dataset preprocessing details
- Experimental setup
- Comparative analysis
- Qualitative visualizations
- Discussion and conclusionsse

For academic use only.
