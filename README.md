# 🔍 AI-Generated Face Detection

> 🚧 **This project is currently in progress.**

A deep learning project to detect AI-generated face images across multiple generation methods.
Core focus: **generalization performance** across unseen generation sources — not just accuracy.

## 🎯 Objectives

- Detect AI-generated faces using **EfficientNetV2S** on a large-scale dataset (~316K images)
- Analyze **ID / OOD generalization** across 3 different fake generation sources
- Interpret model decisions via **Grad-CAM** and deploy a **Streamlit demo**

## 📦 Dataset & Model

**~316K images** | Real 158K / Fake 158K | Equal sampling per source to prevent bias

| # | Dataset | Fake Generation Method |
|---|---------|----------------------|
| 1 | GRAVEX-200K | OpenCV GAN |
| 2 | 140k Real and Fake Faces | StyleGAN3 |
| 3 | 130k Real vs Fake Face | Flux.dev / Flux.pro / SDXL |

**EfficientNetV2S** (ImageNet pretrained) · Mixed Precision · Phase 1: Head Training → Phase 2: Partial Fine-tuning

## 📊 Current Results (Phase 1)

| Metric | Train | Validation |
|--------|-------|------------|
| Accuracy | 76.1% | 70.8% |
| AUC | 0.847 | **0.876** |
| Recall (real) | 0.769 | 0.944 |

## 🛠️ Tech Stack

`TensorFlow` `EfficientNetV2S` `tf.data` `Grad-CAM` `Streamlit` `Python 3.10` `CUDA 12.6` `RTX 3060`

## ✅ Progress

- [x] Dataset collection & preprocessing (~316K images)
- [x] Automated data split pipeline (`prepare_dataset.py`)
- [x] Phase 1 Head Training (Val AUC 0.876)
- [ ] Phase 2 Partial Fine-tuning
- [ ] Per-source ID / OOD evaluation
- [ ] Grad-CAM visualization
- [ ] Streamlit demo deployment

> Baseline reference: [Kaggle - Fake Face Detection with Keras (Acc 0.987)](https://www.kaggle.com/code/naghasnasrollahi/fake-face-detection-with-keras-accuracy-0-987) · MIT License
