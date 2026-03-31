# Visual Recognition Mini Project - Part 1  
## Multi-Object Apparel Detection and Instance Segmentation

---

## 👥 Team Members

- Saheem Showkat Reshi (IMT2023051)  
- Ayush Mishra (IMT2023129)  
- Harsh Sinha (IMT2023571)  
- Nipun Verma (IMT2023591)

---

## 🔗 Resources

- [GitHub Repository](https://github.com/NipunVermaiiitb/Visual-Recognition--1)  
- [Hugging Face Model](https://huggingface.co/1LeoMessi0/VRMP1_IMT2023591_IMT2023571_IMT2023129_IMT2023051/tree/72adb68fcd5080503fe7b389b782fe8d7da41dc3)  
- [Google Drive (All Models)](https://drive.google.com/drive/folders/1YyAEKo0mRQHuEjxIVeRvL_7YsI8FL-Nk?usp=sharing)

---

## 📌 Introduction

With the growth of fashion e-commerce, automated systems for clothing understanding are increasingly important. This project focuses on building a robust visual recognition pipeline using the DeepFashion2 dataset.

We address two main tasks:

- **Multi-label Classification:** Predict multiple clothing categories present in an image  
- **Detection & Segmentation:** Localize objects and generate pixel-level masks  

Challenges include:
- Class imbalance  
- Occlusion and overlap  
- Limited compute  

---

## ⚙️ Implementation Notes

### Transfer Learning Strategy

- **Basic TL (Frozen Backbone):**
  - Poor performance due to lack of domain adaptation

- **TL + Fine-Tuning (Final Approach):**
  - Partial unfreezing → gradual learning  
  - Full fine-tuning → better adaptation  
  - Lower LR in later stages  
  - Multi-label setup using `BCEWithLogitsLoss`  

---

## 📊 Dataset Description & EDA

### Dataset Statistics

- Total Images: **191,961**
- Total Objects: **312,186**
- Avg Objects/Image: **1.63**

### Class Distribution (Top 10)

| Category | Count |
|----------|------|
| Short Sleeve Top | 71,645 |
| Trousers | 55,387 |
| Shorts | 36,616 |
| Long Sleeve Top | 36,064 |
| Skirt | 30,835 |
| Vest Dress | 17,949 |
| Short Sleeve Dress | 17,211 |
| Vest | 16,095 |
| Long Sleeve Outwear | 13,457 |
| Long Sleeve Dress | 7,907 |

### Key Observations

- Strong **class imbalance**
- Most images contain **1–2 objects**
- Minority classes are harder to learn

---

## 🧹 Data Preprocessing

### Augmentations

- Random Horizontal Flip  
- Rotation (10–15°)  
- Color Jitter  
- Random Crop  

### Final Pipeline

**Train:**
- Resize → 256×256  
- Random Crop → 224×224  
- Flip  
- Normalize  

**Validation:**
- Resize → 256×256  
- Center Crop → 224×224  
- Normalize  

---

## 🧪 Experimental Setup

### Hardware

- Kaggle GPU (T4x2)

---

### Classification Models

- ResNet-50  
- EfficientNet-B0  
- MobileNetV3  

All use:
- Multi-label setup  
- BCEWithLogitsLoss  

---

### Detection & Segmentation Models

- YOLOv8 (segmentation)  
- Mask R-CNN  
- U-Net  

---

## 📈 Results

### Classification (Overall)

| Model | Training | Precision | Recall | Macro F1 | Micro F1 |
|------|--------|----------|--------|----------|----------|
| ResNet50 | Scratch | 0.7897 | 0.7317 | 0.7596 | 0.7389 |
| ResNet50 | TL | 0.8679 | 0.8263 | 0.8465 | 0.8590 |
| EfficientNet | TL | **0.8763** | 0.8516 | **0.8515** | 0.8638 |

---

### Detection

| Model | Training | mAP | F1 |
|------|--------|-----|----|
| YOLO | Scratch | 0.532 | 0.547 |
| YOLO | TL | **0.7012** | **0.8079** |
| Mask R-CNN | TL | 0.011 | 0.6092 |

---

### Segmentation

| Model | Training | mIoU | Dice |
|------|--------|------|------|
| YOLO | TL | **0.6583** | **0.7222** |
| Mask R-CNN | TL | 0.4454 | 0.5687 |
| U-Net | TL | 0.5257 | 0.5794 |

---

## 📊 Analysis

### Classification

- Transfer learning improves all models  
- EfficientNet performs best  
- MobileNet offers efficiency tradeoff  

---

### Detection

- YOLO clearly outperforms Mask R-CNN  
- Mask R-CNN suffers from poor localization  

---

### Segmentation

- YOLO gives best overall performance  
- U-Net improves with TL but lacks instance separation  
- Mask R-CNN unstable  

---

## ⚠️ Failure Cases

- Overlapping objects → merged masks  
- Class confusion (short vs long sleeve)  
- Poor minority class performance  
- Small object detection issues  
- Coarse segmentation boundaries  

---

## 📚 Key Learnings

- Transfer learning is critical  
- Fine-tuning improves stability  
- Class imbalance significantly affects results  
- Model complexity ≠ better performance  
- YOLO best for detection + segmentation  

---

## ✅ Conclusion

- EfficientNet best for classification  
- YOLO best overall for detection & segmentation  
- U-Net useful but limited  
- Mask R-CNN underperformed due to training constraints  

---

## 🚀 Future Work

- Handle class imbalance better  
- Train Mask R-CNN longer  
- Try advanced architectures  
- Improve segmentation precision  