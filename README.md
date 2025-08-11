# 🐟 Multiclass Fish Image Classification

## 📋 Project Overview

This repository contains code for an end-to-end **multiclass fish image classification** system using deep learning. Multiple state-of-the-art models — including **VGG16**, **ResNet50**, **MobileNet**, **EfficientNetB0**, and **InceptionV3** — are trained and benchmarked to classify fish species from images. The best-performing model is deployed through an interactive **Streamlit web application**.

---

- **Classes:**  
- `animal fish`, `animal fish bass`, `fish sea_food black_sea_sprat`, `fish sea_food gilt_head_bream`, `fish sea_food hourse_mackerel`, `fish sea_food red_mullet`, `fish sea_food red_sea_bream`, `fish sea_food sea_bass`, `fish sea_food shrimp`, `fish sea_food striped_red_mullet`, `fish sea_food trout`
- **Total Images:**  
- **Train:** 6,655  
- **Validation:** 1,092  
- **Test:** 2,987  

---

## ⚙️ Features

- 🧪 **Custom CNN & Transfer Learning**
  - Data preprocessing & augmentation
  - Class imbalance handled with computed class weights
  - Training, evaluation, and visual analysis
- 🔥 **Best Model:**  
  - **InceptionV3**: `99.84% Accuracy`, `0.0101 Loss`
- 🎨 **Streamlit Web App**
  - User uploads a fish image
  - Displays predicted category and confidence scores
  - Shows probability table for all classes

---

## 🚀 Quickstart

1. Clone the repository

    `git clone https://github.com/gvdharun/Multiclass-Fish-Image-Classification.git`

    cd fish-classification

2. Install dependencies

    `pip install -r requirements.txt`

4. Train or download a pretrained model

    `best_inceptionv3_finetuned.h5`

6. Launch Streamlit app

    `streamlit run fish_app.py`

---

## 📁 Repository Structure
```
├──  📁 Dataset/
|        ├── 📁 data/
|           ├── 📁 train/
|           ├── 📁 val/
|           └── 📁 test/ 
| 
├── 📁 models/ # Saved models
|      ├── best_efficientnetb0.h5
|      ├── best_inceptionv3_finetuned.h5
|      ├── best_mobilenetv2.h5
|      ├── best_resnet50_finetune.h5
|      ├── best_vgg16_finetune.h5
| 
├── 📊 Fish_Classification.ipynb       # Jupyter notebook / Transfer learning scripts
├── 📝fish_app.py                     # Streamlit app
├── 📝requirements.txt
└── 📚 README.md
```
---

## 🏆 Results

| Model             | Accuracy (%) | Loss    |
| ----------------- | ----------- | ------- |
| **VGG16**| 99.65   | 0.0106 |
| **ResNet50**| 57.55   | 1.1437 |
| **MobileNetV2**| 99.65   | 0.0094 |
| 🎉 **InceptionV3**| **99.84**   | **0.0101** |
| **EfficientNetB0**| 14.50   | 2.3953 |

---

## 💻 Deployment

- **Interactive Streamlit App!**  
  Run `streamlit run fish_app.py` to predict fish species from uploaded images.

---

## 📌 Conclusion

Robust preprocessing, class balancing, and transfer learning enable deep learning models to classify fish species with high accuracy. The system is user-friendly, reproducible, and deployable for real-world use.

---
