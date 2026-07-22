# 🔍 AI-Generated Face Detection with EfficientNetV2S

Binary real / AI-generated face classification trained on ~400K images across five generation sources.
The goal was never raw accuracy — it was **generalization to generators the model has never seen**.

<br>

## 📌 Why this matters

In a PNAS study (Nightingale & Farid, 2022, n=315), people asked to tell AI faces from real ones scored **48.2%** — statistically indistinguishable from a coin flip.
Meanwhile reported deepfake fraud losses went from $128M (2020–2023 combined) to $360M (2024) to **$1.1B (2025)** — roughly 7× in a single year.

Human eyes no longer work as a filter. That's the gap this model targets.

<br>

## 🗂️ Dataset

Screened 8 Kaggle candidates, kept **5** after filtering for resolution consistency, aspect ratio, and format. All images resized to **256×256**, `fake : real = 1 : 1`.

| Source | Role | Generator |
|---|---|---|
| [GRAVEX-200K](https://www.kaggle.com/datasets/muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal) | fake + real | OpenCV-based GAN |
| [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) | fake + real | StyleGAN3 |
| [1 Million Fake Faces](https://www.kaggle.com/datasets/tunguz/1-million-fake-faces) | fake | StyleGAN |
| [130k Real vs Fake Face](https://www.kaggle.com/datasets/shreyanshpatel1/130k-real-vs-fake-face) | fake + real | FLUX.dev · FLUX.pro · SDXL |
| [FFHQ](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq) | real | — |

**Total: 396,597 images.** A late expansion added 80,000 more (40K StyleGAN fakes + 40K FFHQ reals) via a shuffled, duplicate-excluded split script — every source-test `real` is disjoint from train/valid to avoid leakage.

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

Two-phase transfer learning:

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

## 📊 Results

Final model, threshold 0.24, applied identically to every test set:

| Test set | Type | Images | Accuracy | AUC | Macro F1 |
|---|---|---|---|---|---|
| Combined ID test | ID | 26,876 | 97.8% | 0.998 | 0.978 |
| StyleGAN | ID | 20,000 | 99.5% | 0.9999 | 0.995 |
| GRAVEX | ID | 9,200 | 91.7% | 0.984 | 0.917 |
| **FLUX / SDXL** | **OOD** | 9,200 | **99.6%** | **0.9999** | **0.996** |

Phase 2 lifted every source, not just the average:

| Test set | Before fine-tuning | Final |
|---|---|---|
| Combined ID test | 90.8% | **97.8%** |
| StyleGAN | 92.5% | **99.5%** |
| GRAVEX | 83.5% | **91.7%** |
| FLUX / SDXL (OOD) | 96.4% | **99.6%** |

The model holds up on a generator family it never trained on — it learned something closer to "what AI generation looks like" than "what these specific datasets look like."

<br>

## ⚠️ The result we don't fully trust yet

Overfitting is ruled out — a memorizing model would collapse on the OOD set, and this one scores highest there.

But that's exactly the problem: **OOD accuracy (99.6%) exceeding ID accuracy (97.8%) is not normal.** A model should do *worse* on unseen distributions, not better.

The likely explanation is **shortcut learning** — each dataset carries its own fingerprint (compression artifacts, resize history, crop conventions), and the model may be reading the dataset rather than the generator. FLUX/SDXL being the cleanest, most distinctive-looking source would then make it the *easiest* set, not the hardest. GRAVEX sitting 8 points below everything else points the same way.

This is flagged as an open problem rather than a win. Verifying it requires cross-dataset evaluation where real and fake images come from the same processing pipeline.

<br>

## 🖥️ Demo

A Streamlit app takes an uploaded image and returns `P(real)` / `P(fake)` against the configured threshold.

The clearest before/after: a teammate's own selfie was called **AI / FAKE at 72.27%** before Phase 2 fine-tuning, and correctly **REAL at 77.05%** after — on the exact same image.

```bash
streamlit run outputs/<RUN_NAME>/app/streamlit_app.py
```

<br>

## 🛠️ Tech Stack

TensorFlow / Keras · EfficientNetV2S · tf.data · Mixed Precision · scikit-learn · Grad-CAM · Streamlit
Python 3.10 · CUDA 12.6 · RTX 3060

<br>

## 🚧 Limitations & Next Steps

- **Shortcut learning suspected** (see above) — needs cross-dataset validation with matched preprocessing
- **GRAVEX is the weak point at 91.7%** — real/fake balance within that single source is uneven
- **Faces only** — full bodies, backgrounds, and scenes are out of scope

Planned: adding Midjourney and DALL·E families, source-diversified real images for honest cross-dataset evaluation, extending detection from faces to full images, and Grad-CAM visualization with a HuggingFace deployment.

<br>

---

<sub>Team 데릉데릉 (임효진, 김나연, 이제이, 공두현) · BDAS Big Data Analysis Society showcase · Baseline reference: Kaggle *Fake Face Detection with Keras* (Acc 0.987, MIT License)</sub>
