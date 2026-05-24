 # 🔍 AI-Generated Face Detection

> 🚧 **This project is currently in progress.**

A deep learning project to detect AI-generated face images across multiple generation methods (GAN, Diffusion, DeepFake).
The core focus is **generalization performance analysis** — not just accuracy.

---

## 🎯 Objectives

- Build an AI-generated face detector based on **EfficientNetV2S**
- Construct a large-scale dataset (~316K images) covering 3 different generation methods
- Analyze **ID / OOD generalization performance** across generation sources
- Interpret model decisions via **Grad-CAM visualization**
- Deploy a **Streamlit web demo**

---

## 🏗️ Project Pipeline

```
Reference: Kaggle DenseNet baseline
→ Upgrade to EfficientNetV2S
→ Build diverse dataset (~316K images, 3 fake sources)
→ Phase 1: Head Training (backbone frozen)
→ Phase 2: Partial Fine-tuning
→ ID / OOD evaluation by source
→ Streamlit demo deployment
```

---

## 📦 Dataset

Total ~**316,000 images** (Real 158,000 / Fake 158,000)

| # | Dataset | Fake Generation Method | Size |
|---|---------|----------------------|------|
| 1 | GRAVEX-200K | OpenCV GAN | 100K each |
| 2 | 140k Real and Fake Faces | StyleGAN3 | 70K each |
| 3 | 130k Real vs Fake Face | Flux.dev / Flux.pro / SDXL | Fake 64K / Real 70K |

**Data Split Strategy**
- Equal sampling per source (46K each) to prevent bias toward any single generation method
- Train 80% / Valid 10% / Test 10%
- Separate `test_by_source/` folder for per-source OOD evaluation

---

## 🧠 Model

**EfficientNetV2S** (ImageNet pretrained)

```
Input 256×256 RGB
→ EfficientNetV2S backbone (include_preprocessing=True)
→ GlobalAveragePooling2D
→ Dropout(0.35)
→ Dense(1, sigmoid, float32)
```

**Training Strategy**
- Phase 1: Backbone frozen → Head only training
- Phase 2: Partial fine-tuning on last 60 layers (LR=1e-5)
- Mixed Precision (float16) — optimized for RTX 3060
- EarlyStopping / ReduceLROnPlateau / ModelCheckpoint

---

## 📊 Current Results (Phase 1)

| Metric | Train | Validation |
|--------|-------|------------|
| Accuracy | 76.1% | 70.8% |
| AUC | 0.847 | **0.876** |
| Recall (real) | 0.769 | 0.944 |

> Phase 2 fine-tuning and OOD analysis in progress

---

## 🔬 Evaluation Design

- **ID Test**: Combined independent test set from all 3 sources
- **Source Test**: Per-source evaluation (GRAVEX / StyleGAN / Flux·SDXL)
- **OOD Analysis**: Generalization performance on unseen generation sources
- **Metrics**: Accuracy / Precision / Recall / F1 / AUC / Confusion Matrix / ROC Curve

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Framework | TensorFlow / Keras |
| Model | EfficientNetV2S (ImageNet pretrained) |
| Data Pipeline | tf.data + prefetch (AUTOTUNE) |
| Visualization | Grad-CAM, Confusion Matrix, ROC Curve |
| Demo | Streamlit |
| Environment | Python 3.10, CUDA 12.6, RTX 3060 12GB |

---

## 📁 Key Files

```
├── efficientnetv2s_fake_face_FINAL.ipynb  # Main training notebook
├── prepare_dataset.py                      # Automated dataset split script
└── outputs/
    └── efficientnetv2s_experiment_01/
        ├── models/                         # Saved weights
        ├── results/                        # Evaluation CSVs & plots
        ├── gradcam/                        # Grad-CAM outputs
        └── app/                            # Streamlit app
```

---

## ✅ Progress

- [x] Dataset collection & preprocessing (~316K images)
- [x] Automated data split pipeline (`prepare_dataset.py`)
- [x] EfficientNetV2S model & training pipeline
- [x] Phase 1 Head Training (Val AUC 0.876)
- [ ] Phase 2 Partial Fine-tuning
- [ ] Per-source ID / OOD evaluation
- [ ] Grad-CAM visualization
- [ ] Streamlit demo deployment

---

## 💡 Key Contributions

1. **Generalization-focused evaluation**: Quantitative OOD analysis per generation source — not just overall accuracy
2. **Diverse large-scale dataset**: Equal mix of 3 generation methods (OpenCV GAN / StyleGAN3 / Diffusion) to reduce source-specific overfitting
3. **End-to-end reproducible pipeline**: Automated data preparation, training, evaluation, and demo

---

## 📌 Reference

- Baseline: [Kaggle - Fake Face Detection with Keras (Acc 0.987)](https://www.kaggle.com/code/naghasnasrollahi/fake-face-detection-with-keras-accuracy-0-987)
- License: MIT
