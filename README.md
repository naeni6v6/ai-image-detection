<h1 align="center">🔍 Fakit — AI-Generated Face Detection</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Model-EfficientNetV2S-0066CC?style=for-the-badge" alt="EfficientNetV2S">
  <img src="https://img.shields.io/badge/Accuracy-95.2%25-success?style=for-the-badge" alt="Accuracy 95.2%">
  <img src="https://img.shields.io/badge/Unseen_Diffusion_AUC-0.935-success?style=for-the-badge" alt="Unseen diffusion AUC 0.935">
  <img src="https://img.shields.io/badge/Video_Accuracy-80.3%25-success?style=for-the-badge" alt="Video accuracy 80.3%">
</p>
<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/timm-000000?style=flat" alt="timm">
  <img src="https://img.shields.io/badge/ONNX_Runtime_Web-005CED?logo=onnx&logoColor=white" alt="ONNX Runtime Web">
  <img src="https://img.shields.io/badge/Chrome_Extension-4285F4?logo=googlechrome&logoColor=white" alt="Chrome Extension">
  <img src="https://img.shields.io/badge/images-396K-lightgrey" alt="396K images">
</p>
<p align="center">
  Detecting AI-generated faces in photos and videos — from a model that scored suspiciously well, to one that
  generalizes to generators it has never seen, shipped as the web service <a href="https://fakit-six.vercel.app/"><b>Fakit</b></a>.
</p>

<br>

## 📌 Overview

AI-generated faces are now hard to tell apart from real photos, and they are increasingly used for impersonation and disinformation.
This project builds a binary **REAL / FAKE** face classifier on **EfficientNetV2S** and deploys it as a website where anyone can check
an image or video — with Grad-CAM showing *why* the model decided.

The focus was not a high headline score, but a score that **can be trusted**: removing leakage, removing shortcuts,
and testing on generators the model never trained on.

<br>

## 🗂️ Dataset

- **396,597 face images** merged from **8 public Kaggle datasets**
- Fakes labeled into 3 families — **GAN · Diffusion · partial manipulation** (face swap, expression edit) — to enable family-level generalization tests
- Real images diversified with FFHQ and video frames · `FAKE : REAL = 1 : 1`

<br>

## 🔎 Finding the Real Performance

**1. AUC 0.99 was data leakage**
The first model scored AUC 0.99 — too good to trust. Public datasets turned out to share the same images:
**~128K duplicates** landed in both train and test. Removed them with **dHash** near-duplicate detection and re-split train / valid / test.

**2. The model was reading the file, not the face**
Using only file metadata (PNG or not, JPEG quantization table, bytes per pixel) — without looking at the image at all —
real / fake could be classified with up to **90%** accuracy. A shortcut.
→ **JPEG re-encoding + face crop** remove format and framing cues so the model judges the face itself.

**3. Seen generators ≠ real accuracy**
Leave-One-Generator-family-Out (LOGO): train without one family (GAN / Diffusion / manipulation), test only on it.
Seen families scored **96%+**, but unseen families fell to **~50%** — a coin flip.

<br>

## 🧠 Model

Instead of memorizing each generator's fingerprint, the model is taught traces common to *any* generator using **pseudo-fakes** made from real photos:

| Pseudo-fake | How | What it learns |
|---|---|---|
| **VAE** | Reconstruct a real photo through a VAE | Decoding traces of generative models |
| **SBI** | Blend a real face with a transformed copy of itself | Blending-boundary traces |

- Base: **EfficientNetV2S** + VAE · SBI pseudo-fakes + downscale / blur augmentation
- Training tuned with **input 320** and **EMA**
- Threshold: **P(real) 0.30** · exported to **ONNX** and run in the browser

```
image → BlazeFace face detection → 2× crop → 256×256 preprocessing → EfficientNetV2S → REAL / FAKE
```

<br>

## 📊 Results

<table>
<tr>
<td valign="top">

**Unseen Diffusion generators (LOGO AUC)**

| Training | AUC |
|---|---|
| Base | 0.728 |
| + VAE | 0.876 |
| **+ VAE + SBI** | **0.935** |

Stable Diffusion · SDXL · FLUX Dev · FLUX Pro — all **AUC 0.92+**

<br>

**Final model**

| Metric | Score |
|---|---|
| Accuracy (independent 20,000 images) | **95.2%** |
| Unseen Diffusion AUC | **0.935** |

</td>
<td valign="top">
<img src="https://raw.githubusercontent.com/naeni6v6/ai-image-detection/develop/results/presentation_charts/1_unseen_diffusion_auc.png" alt="Unseen diffusion AUC" width="520">
</td>
</tr>
</table>

<br>

## 🎞️ Video Extension

- MP4 · WebM, 3–40 s, up to 90 MB · **16 / 32 / 48 / 64 frames** sampled by length
- Frozen EfficientNetV2S extracts per-frame features and AI scores
- Mean · std · frame-to-frame change → **2 RBF classifiers** → segment scores merged into one video verdict

| Test videos | Accuracy | F1-score |
|---|---|---|
| 1,000 | **80.3%** | **80.2%** |

<br>

## 🖥️ Fakit Web Service

**🔗 https://fakit-six.vercel.app/**

| Input | Analysis | Explanation | Use |
|---|---|---|---|
| Multi-upload · drag · paste · link | In-browser ONNX inference, per-face verdict | Grad-CAM · threshold · result guide | History · re-check · Chrome extension |

<br>

## ⚠️ Limitations & Future Work

- Partial manipulation (face swap, expression edit) is harder to catch — **recall ~77%**
  → add more manipulation data, extend validation to newer generators and real-world images
- SBI improves generalization but raises false positives on **retouched / beauty-filtered real photos**
  → include retouched photos as REAL in training

<br>

## 🛠️ Tech Stack

Python · PyTorch · timm · EfficientNetV2S · BlazeFace · Grad-CAM · ONNX Runtime Web · Chrome Extension · Vercel

<br>

---

<sub>BDAS Big Data Analysis Society · Team <b>데릉데릉</b> — 임효진 · 김나연 · 공두현 · 이제이 · Advisor: Prof. 남진현 · 2026.03 – 2026.11</sub>
