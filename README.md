🔍 Fakit — AI-Generated Face Detection
 
 
 

 
 
 
 
 

Telling real faces from AI-generated ones in photos and videos — evolved from a Keras baseline (v1) whose suspiciously perfect score turned out to be a shortcut, into a face-crop pipeline (v2) evaluated on generators it has never seen, and shipped as a website and Chrome extension that run entirely in the browser.


📌 Overview
Humans score 48.2% at telling real faces from AI-generated ones — no better than a coin flip (Nightingale & Farid, PNAS 2022). So the goal was never raw accuracy — it was generalization to generators the model has never seen, and robustness to the compression and resizing that happen once a photo is posted online.

What	How
📷 Photo check	Detect and crop each face → EfficientNetV2-S REAL / FAKE verdict + Grad-CAM evidence map
🎞️ Video check	Sample 16 / 32 / 48 / 64 frames by clip length → video-level classifier
🧩 Chrome extension	REAL / FAKE badges on face images in the page you're browsing (e.g. Instagram)
🔬 Research	Unseen-generator generalization (LOGO), SNS-style degradation, false positives on retouched photos
All inference runs inside the browser (ONNX Runtime Web, TensorFlow.js) — images and videos are never uploaded to a server.


v1 — Keras Baseline

Dataset
Screened 8 Kaggle candidates and kept 5 after filtering for resolution consistency, aspect ratio, and format. All images resized to 256×256, fake : real = 1 : 1 — 396,597 images in total.

Source	Role	Generator
GRAVEX-200K	fake + real	OpenCV-based GAN
140k Real and Fake Faces	fake + real	StyleGAN
1 Million Fake Faces	fake	StyleGAN
130k Real vs Fake Face	fake + real	FLUX.dev · FLUX.pro · SDXL
FFHQ	real	—
FLUX / SDXL was deliberately excluded from training so it could serve as an out-of-distribution (OOD) test set.


Model
TensorFlow / Keras EfficientNetV2S, chosen over DenseNet121 for accuracy-per-FLOP and sensitivity to fine-grained texture
Two-phase transfer learning: frozen-backbone head training (val AUC 0.9907) → partial fine-tuning of the last 48 layers, BN frozen (val AUC 0.9982)
Weak augmentation only — strong augmentation risks erasing the very artifacts being detected
Decision threshold 0.24, chosen by macro-F1 on validation only (tuning on test is leakage)

📊 Results
Test set	Type	Images	Accuracy	AUC
Combined ID test	ID	26,876	97.8%	0.998
StyleGAN	ID	20,000	99.5%	0.9999
GRAVEX	ID	9,200	91.7%	0.984
FLUX / SDXL	OOD	9,200	99.6%	0.9999

⚠️ The result we didn't trust — and why
OOD accuracy (99.6%) beating ID accuracy (97.8%) is not normal. A model should do worse on generators it has never seen. The suspicion was shortcut learning: the model reading the dataset rather than the generator.

This was then confirmed. A classifier given only the face size / framing — no pixels of the face at all — separates real from fake on the FLUX / SDXL set with AUC 0.990 (accuracy 96.9%), because the AI faces in that source are framed much smaller (mean face-size ratio 0.36 vs 0.60 for real photos). On StyleGAN the same trick scores 0.52 — chance level.

Framing-only classifier	Accuracy	AUC
FLUX / SDXL	96.9%	0.990
GRAVEX	58.2%	0.601
StyleGAN	51.6%	0.521
v1's "best" number was mostly composition, not generation artifacts. That motivated v2.


v2 — Final Model & Fakit

What changed
Face crop first: BlazeFace detection → 2× context square crop, so every image is judged on the face, not the framing
Broader fakes: three families — GAN (StyleGAN …), diffusion (FLUX, SD, SDXL …), and face manipulation — against real faces (FFHQ …)
Pseudo-fakes: real photos reconstructed through the Stable Diffusion VAE are added as fakes, teaching diffusion-style traces without a specific generator
Degradation augmentation: downscaling (0.4–1.0×), blur, and JPEG compression, so photos survive being posted to social media
Honest evaluation: Leave-One-Generator-family-Out (LOGO) — train without a whole family, test only on it

Model
Framework: PyTorch (migrated from Keras/TensorFlow), exported to ONNX for the browser
Architecture: EfficientNetV2-S, ImageNet-21k pretrained (tf_efficientnetv2_s.in21k_ft_in1k, timm)
Tuned on validation only (t0 → t6): 8 epochs, EMA 0.999, drop path 0.2, 50% face-crop mixing, 320×320 input
Thresholds: 0.34 on face crops / 0.52 on full images (validation macro-F1)
photo → BlazeFace face detection → 2× context square crop → 256×256
      → JPEG q90 re-encode → normalize (mean/std 0.5) → upscale to 320×320
      → EfficientNetV2-S → P(real) ≥ 0.34 → REAL, else AI
If no face is found, the verdict is "undetermined" rather than a guess.


📊 Results (face crop)
Test set	AUC	Accuracy	Fake recall	Real recall
Same-distribution test (6,000)	0.998	97.6%	0.965	0.988
Same test, JPEG q50	0.997	96.9%	0.971	0.967
Held-out extra test (20,000)	0.994	95.5%	0.937	0.973
SNS robustness (extra test AUC): original 0.994 · JPEG 50 0.989 · 0.5× downscale 0.992 · blur σ1.0 0.993

vs. the previous photo model (ConvNeXt-T), same recipe:

ConvNeXt-T	EfficientNetV2-S
AUC	0.990	0.994
Balanced accuracy	94.6%	95.5%
Real recall	95.4%	97.3%
JPEG 50 balanced acc.	91.7%	93.2%
Browser model size	56 MB	42 MB


🔬 Generalization to unseen generators (LOGO, OOD AUC)
Pseudo-fake	Diffusion	GAN	Manipulation
Baseline (untuned)	0.728	0.473	0.458
None	0.722	0.489	0.393
VAE (deployed)	0.876	0.530	0.429
SBI	0.907	0.383	0.527
VAE + SBI	0.935	0.466	0.507

VAE pseudo-fakes lifted unseen diffusion AUC from 0.73 → 0.88
Unseen GAN and manipulation families stay near 0.5 — the model is still weak on a brand-new family
High AUC ≠ caught: at the deployed threshold, fake recall on unseen SDXL is only ~0.11, so AUC and recall are reported together

🧪 False positives on retouched real photos
300 real photos with beauty retouching applied → share wrongly flagged as FAKE:

Retouch	Light	Medium	Strong
Skin smoothing	0.0%	4.3%	12.3%
Eye enlarging	0.0%	0.0%	0.0%
Face slimming	0.0%	0.0%	0.3%
Beauty combo	5.7%	14.3%	–
The stronger the skin smoothing, the more often a real face is mistaken for AI.

Face crop matters more than the model: on small faces (200px face in a 900×700 scene, 400 images), judging the full image gave 56% accuracy; detecting and cropping the face first gave 98%.


🖥️ Fakit — Website & Chrome Extension
🔗 https://fakit-lens.vercel.app

Photo check — per-face REAL score and verdict, batch upload, local history
Video check — 3–40 s MP4 / WebM analyzed in the browser, with per-segment scores and Grad-CAM on key frames
Grad-CAM — highlights the face regions the model relied on (evidence, not proof of where manipulation happened)
Chrome extension (AI Face Detector) — an AI face scan button badges every face image on the page REAL 94% / FAKE 87% / undetermined; the model ships inside the extension, so nothing leaves the browser
Video model: frames are sampled by length (≤10 s: 16 · ≤20 s: 32 · ≤30 s: 48 · longer: 64), EfficientNetV2-S features are pooled into 772-dim temporal statistics, and two RBF classifiers are averaged. On 1,000 test videos (500 real · 500 Sora) accuracy is about 80%, with ~19% of real videos flagged as AI — treat video results as a reference only.


📂 Repository
Branch	Contents
main	Project overview (this document), v1 notebook, Streamlit demo, final presentation
develop	Final model, FastAPI local inference server, Chrome extension, ONNX conversion, experiment results
fakit-preview	Deployed website (Fakit)
study/KongDuHyun	Video detection model
feature/generalization-study	Second-round generalization study (model comparison, LOGO)
develop
├── server/        FastAPI local server (face detection, prediction, Grad-CAM, ELA)
├── extension/     Chrome extension (Manifest V3, in-browser inference)
├── web_model/     Final model → ONNX conversion & verification
├── experiments/   Retouched-photo experiments
└── results/
    ├── phase9_efficientnet/   Final model tuning · LOGO · robustness
    ├── experiment_b/          Retouch false positives
    ├── framing_shortcut/      v1 shortcut-learning check
    └── presentation_charts/   Presentation figures
Model weights (.pt, .onnx) are not committed due to size.


🛠️ Tech Stack
Python, PyTorch, timm (v2) · TensorFlow / Keras (v1)
EfficientNetV2-S, BlazeFace, Stable Diffusion VAE pseudo-fakes, SBI, EMA, Grad-CAM
ONNX Runtime Web, TensorFlow.js, Chrome Extension (Manifest V3), FastAPI, Streamlit, Vercel

⚠️ Limitations & Future Work
Generalization to unseen GAN and face-manipulation families is still weak (LOGO AUC around 0.4–0.5)
Real photos with strong skin smoothing or beauty filters can be flagged as AI
No verdict when the face is missing or too small
Scores are raw model outputs, not calibrated probabilities; Grad-CAM shows evidence, not where manipulation occurred
Next: pseudo-fakes for GAN / manipulation traces, threshold calibration, and a stronger temporal model for video

BDAS Big Data Analysis Society showcase · Team project with hyojin0226, leejei36 and Kong DuHyun · Baseline reference: Kaggle Fake Face Detection with Keras (Acc 0.987, MIT License)
