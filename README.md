# 🔍 AI-Generated Face Detection

<p align="center">
  <img src="https://img.shields.io/badge/Model-EfficientNetV2S-0066CC?style=for-the-badge" alt="EfficientNetV2S">
  <img src="https://img.shields.io/badge/OOD_Accuracy-99.6%25-success?style=for-the-badge" alt="OOD 99.6%">
  <img src="https://img.shields.io/badge/AUC-0.9983-success?style=for-the-badge" alt="AUC 0.9983">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python 3.10">
  <img src="https://img.shields.io/badge/images-396K-lightgrey" alt="396K images">
</p>

Real vs AI-generated face classification with **EfficientNetV2S**, trained on ~400K images across five generation sources.

Humans score **48.2%** at this task — indistinguishable from a coin flip (Nightingale & Farid, PNAS 2022).
So the goal was never raw accuracy — it was **generalization to generators the model has never seen**.

<br>

## 📊 Results

Final model, decision threshold **0.24**, applied identically to every test set.

| Test set | Type | Images | Accuracy | AUC | Macro F1 |
|---|---|---|---|---|---|
| Combined ID test | ID | 26,876 | 97.8% | 0.9983 | 0.978 |
| StyleGAN | ID | 20,000 | 99.5% | 0.9999 | 0.995 |
| GRAVEX | ID | 9,200 | 91.7% | 0.9836 | 0.917 |
| **FLUX / SDXL** | **OOD** | 9,200 | **99.6%** | **0.9999** | **0.996** |

Phase 2 fine-tuning lifted every source, not just the average:

| Test set | Before fine-tuning | Final |
|---|---|---|
| Combined ID test | 90.8% | **97.8%** |
| StyleGAN | 92.5% | **99.5%** |
| GRAVEX | 83.5% | **91.7%** |
| FLUX / SDXL (OOD) | 96.4% | **99.6%** |

<details>
<summary>Full per-class metrics & curves</summary>

<br>

| Test set | Fake precision | Fake recall | Real precision | Real recall |
|---|---|---|---|---|
| Combined ID test | 0.991 | 0.965 | 0.966 | 0.991 |
| StyleGAN | 0.995 | 0.995 | 0.995 | 0.995 |
| GRAVEX | 0.980 | 0.852 | 0.869 | 0.982 |
| FLUX / SDXL (OOD) | 0.995 | 0.996 | 0.996 | 0.995 |

GRAVEX fake recall (0.852) is the weakest number in the project — roughly 15% of GRAVEX-generated faces slip through as real.

![roc curve](assets/roc_curve.png)

![training curves](assets/training_curves.png)

<sub>Phase 1 head training (epochs 1–15) → Phase 2 fine-tuning (16–25) · val_auc 0.9907 → 0.9982</sub>

</details>

**Baseline reference:** the Kaggle notebook *Fake Face Detection with Keras* (DenseNet121, Acc 0.987, MIT License) was reproduced as a starting point. It trains on a single dataset, so the numbers aren't directly comparable — this project's contribution is the multi-source ID/OOD evaluation split rather than a higher headline accuracy.

<br>

## 🗂️ Dataset

Screened 8 Kaggle candidates, kept **5** after filtering for resolution consistency, aspect ratio, and format. All images resized to **256×256**, `fake : real = 1 : 1`, **396,597 images total**.

| Source | Role | Generator |
|---|---|---|
| [GRAVEX-200K](https://www.kaggle.com/datasets/muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal) | fake + real | OpenCV-based GAN |
| [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) | fake + real | StyleGAN3 |
| [1 Million Fake Faces](https://www.kaggle.com/datasets/tunguz/1-million-fake-faces) | fake | StyleGAN |
| [130k Real vs Fake Face](https://www.kaggle.com/datasets/shreyanshpatel1/130k-real-vs-fake-face) | fake + real | FLUX.dev · FLUX.pro · SDXL |
| [FFHQ](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq) | real | — |

*Dropped:* Human Faces (non-square aspect ratio), Celeb-DF v2 (video only), AI-Face-Dataset-3000 (resolution mismatch).

**The evaluation split is the point.** Beyond a combined ID test set, each generation source gets its own test folder — and **FLUX/SDXL was deliberately excluded from training** so it functions as a true OOD set.

```
data/
├── train/ valid/ test/          # fake/ real/
└── test_by_source/
    ├── gravex/     (9,200)      # ID
    ├── stylegan/  (20,000)      # ID
    └── flux_sdxl/  (9,200)      # OOD — never seen in training
```

<br>

## 🧠 Model & Training

Compared VGGNet · ResNet · DenseNet · EfficientNet · ViT · Xception, then narrowed to DenseNet121 vs EfficientNet.
**EfficientNetV2S** won on the accuracy-per-FLOP tradeoff and on its sensitivity to fine-grained texture — which is exactly what separates a generated face from a photographed one.

| Phase | Backbone | LR | Epochs | val_auc |
|---|---|---|---|---|
| 1 — Head training | frozen | `1e-3` → `5e-4` @ep11 | 15 | 0.9907 |
| 2 — Partial fine-tuning | last 48 layers (BN frozen) | `1e-5` | 10 | **0.9982** |

- Input 256×256 RGB, `include_preprocessing=True` — pixels stay in [0, 255], no manual `/255.`
- Head: GlobalAveragePooling → Dropout 0.35 → Dense(1, sigmoid)
- Mixed precision (`float16`), `tf.data` with prefetch, weak augmentation only (flip, ±2° rotation, 5% zoom, 3% shift, 10% contrast) — strong augmentation risks erasing the very artifacts being detected
- Checkpoint on `val_auc`, EarlyStopping (patience 3, restore best), ReduceLROnPlateau, BackupAndRestore

**Decision threshold = 0.24**, chosen by maximizing macro-F1 **on validation, never on test** — tuning a threshold on the test set is leakage. At this threshold a "fake" verdict requires `P(fake) > 0.76`, which cuts false accusations against real photos.

<br>

## ⚠️ One result we don't fully trust

**OOD accuracy (99.6%) exceeding ID accuracy (97.8%) is not normal** — a model should do worse on unseen distributions, not better. Overfitting is ruled out (a memorizing model would collapse on OOD), which makes it more suspicious, not less.

The likely cause is **shortcut learning**: each dataset carries its own fingerprint — compression artifacts, resize history, crop conventions — so the model may be reading the *dataset* rather than the *generator*. GRAVEX sitting 8 points below everything else points the same way.

Flagged as an open problem, not a win. Verifying it needs cross-dataset evaluation where real and fake images share the same processing pipeline.

<br>

## 🖥️ Web Demo

A **Streamlit** web app was built for live inference — upload a face image and it returns `P(real)` / `P(fake)` against the configured threshold, reading the exported `.keras` model and `inference_config.json` so the app always uses the same threshold as evaluation.

![streamlit demo](assets/demo.png)

<sub>The same selfie: called AI / FAKE at 72.27% before Phase 2 fine-tuning, correctly REAL at 77.05% after.</sub>

```bash
streamlit run app/streamlit_app.py
```

The current app is a functional prototype. A properly designed web interface and public deployment are planned as the next step.

<br>

## 🛠️ Tech Stack

TensorFlow / Keras · EfficientNetV2S · tf.data · Mixed Precision · scikit-learn · Streamlit
Python 3.10 · CUDA 12.6 · RTX 3060

<br>

## 🚧 Limitations & Next Steps

- **Shortcut learning suspected** (see above) — needs cross-dataset validation with matched preprocessing
- **GRAVEX is the weak point at 91.7%**, with fake recall at 0.852 — real/fake balance within that single source is uneven
- **Faces only** — full bodies, backgrounds, and scenes are out of scope

Planned: adding Midjourney and DALL·E families, source-diversified real images for honest cross-dataset evaluation, extending detection from faces to full images, Grad-CAM visualization of model attention, and a redesigned web app with public deployment.

<br>

---

<sub>Team 데릉데릉 (임효진, 김나연, 이제이, 공두현) · BDAS Big Data Analysis Society showcase</sub>
