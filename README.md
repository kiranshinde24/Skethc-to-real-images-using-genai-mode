# Forensic Sketch to Image Generation using Generative AI
## Overview

In many criminal investigations a hand-drawn sketch from an eyewitness is the only available clue, but sketches are imprecise and slow to interpret. This project converts face sketches into realistic face images with generative AI, and compares four different model families on the same paired sketch-photo dataset to find which works best for forensic use.

| Model | Approach | Notebook |
|---|---|---|
| **Pix2Pix** | Conditional GAN (U-Net generator + PatchGAN discriminator) | `SketchtoImage_pix2pix.ipynb` |
| **Diffusion** | Conditional DDPM (U-Net, sketch concatenated with noisy image) | `diffusion.ipynb` |
| **Vision Transformer** | ViT regressor trained from scratch | `transformer.ipynb` |
| **VAE** | Convolutional VAE (sketch -> latent -> photo) | `VAE_ForensicSketchToRealisticImage.ipynb` |
| **Demo UI** | Gradio app to upload a sketch and compare model outputs | `Frontend.ipynb` |

All notebooks are written for **Google Colab** (GPU runtime recommended) and save checkpoints to Google Drive.

## Pipeline

```
Input sketch -> Preprocessing (resize, normalize, augment)
             -> Generative model (GAN | VAE | Diffusion | Transformer)
             -> Generated face -> Evaluation (PSNR, SSIM, LPIPS, CLIPScore) -> Comparison
```

## Dataset

[Person Face Sketches](https://www.kaggle.com/datasets/almightyj/person-face-sketches) (Kaggle, `almightyj/person-face-sketches`): 21,000+ paired sketch-photo samples derived from CelebAMask-HQ, with sketches produced by edge detection and simplification.

```
dataset/
├── train/ { sketches/, photos/ }
├── val/   { sketches/, photos/ }
└── test/  { sketches/, photos/ }
```

Sketch and photo files must pair up one-to-one by name/order.

The Pix2Pix and Transformer notebooks download the dataset via the Kaggle API (you upload `kaggle.json` when prompted). The Diffusion and VAE notebooks read the dataset as a zip from Google Drive; update the zip path at the top of each notebook.

## Models

### 1. Pix2Pix (cGAN)
- **Generator:** U-Net with skip connections (6 down / 5 up blocks), 1-channel sketch -> 3-channel photo, `Tanh` output.
- **Discriminator:** PatchGAN on (sketch, photo) pairs.
- **Loss:** adversarial BCE + 100 x L1.
- **Notebook settings:** 128x128, batch 8, Adam (lr 2e-4, betas 0.5/0.999), 100 epochs, random subset of 647 training pairs.
- Logs PSNR / SSIM per epoch and saves `G_epoch*.pth` / `D_epoch*.pth`.

### 2. Diffusion (conditional DDPM)
- `diffusers.UNet2DModel` with 6 input channels (noisy photo 3 + sketch 3), predicts the added noise (MSE).
- `DDPMScheduler`, 500 timesteps.
- **Notebook settings:** 64x64, batch 16, AdamW (lr 1e-4), 150 epochs, mixed precision, gradient clipping.
- Albumentations pipeline; auto-resumes from the latest checkpoint.
- Evaluation: PSNR, SSIM, LPIPS on 5 validation images.

### 3. Vision Transformer
- 16x16 patch embedding (dim 256) + positional embedding, 6 encoder blocks, 8 heads.
- Mean-pooled tokens -> linear decoder -> 3x128x128 image (`Sigmoid`).
- **Notebook settings:** 128x128, batch 32, Adam (lr 1e-4), MSE loss, mixed precision, 100 epochs.
- Tracks PSNR / SSIM (every 2 epochs) and LPIPS (every 5 epochs); supports resume.

### 4. VAE
- Encoder: 4 strided conv layers -> `mu`, `logvar` (latent dim 256). Decoder: FC + 4 transposed conv layers, `Tanh`.
- **Loss:** L1 reconstruction + 1e-4 x KL divergence.
- **Notebook settings:** 128x128, batch 16, Adam (lr 1e-4), mixed precision, 100 epochs.
- Autosave/resume, keeps the last 3 epoch checkpoints, best model by validation SSIM.
- Evaluation: SSIM, PSNR, LPIPS, CLIP similarity, FID (`pytorch-fid`).

### 5. Demo UI (Gradio)
- **Page 1 - Upload & Generate:** upload a sketch, view the result and switch models via a dropdown.
- **Page 2 - Model Comparison:** metric cards, bar charts (Loss, PSNR, SSIM, CLIPScore) and a metrics table.

> **Current limitation:** the UI is a **demo simulation**. It finds the matching ground-truth photo by file name and applies PIL filters to mimic each model's look; it does not yet run the trained checkpoints, and the comparison numbers are hard-coded. To make it real, load the saved `.pth` weights from each notebook and run inference on the uploaded sketch.

## Results (as reported in the project report)

| Model | PSNR (dB) | SSIM | LPIPS | CLIPScore | Loss |
|---|---|---|---|---|---|
| **Pix2Pix GAN** | 30.5 | 0.72 | 0.25 | 0.83 | 0.18 |
| VAE | 24.2 | 0.40 | 0.36 | 0.74 | 0.38 |
| Transformer | 21.8 | 0.44 | 0.42 | 0.69 | 0.31 |
| Diffusion | 19.3 | 0.47 | 0.51 | 0.63 | 0.25 |

Qualitative findings: Pix2Pix gave the sharpest, most identity-preserving faces; VAE outputs were smooth but lost detail; the Transformer captured structure but produced blurry/distorted faces; Diffusion gave detailed textures but was slow to train and sample.

Metric guide: higher is better for PSNR, SSIM, CLIPScore; lower is better for LPIPS, FID and loss.

## Installation

```bash
pip install torch torchvision torchaudio
pip install diffusers==0.28.0 "huggingface_hub==0.25.2" transformers accelerate
pip install albumentations kornia einops
pip install torchmetrics lpips scikit-image pytorch-fid tqdm pillow matplotlib pandas
pip install git+https://github.com/openai/CLIP.git
pip install gradio
```

Tested on Google Colab GPUs; the report's experiments used an NVIDIA RTX 3060 (12 GB) with PyTorch.

## Usage

1. Open the notebook for the model you want in Google Colab and pick a GPU runtime.
2. Mount Google Drive and set the dataset path (or upload `kaggle.json`).
3. Run the cells top to bottom. Checkpoints and sample images are saved to your Drive (`sketch2real_out/`, `pix2pix_fast_checkpoints/`, `vae_sketch2real_checkpoints/`, `vit_sketch2real_checkpoints/`).
4. To resume interrupted training, re-run the notebook; Diffusion and VAE pick up the latest checkpoint automatically.
5. For the demo, run `Frontend.ipynb`; Gradio prints a public share link.

## Project Structure

```
.
├── SketchtoImage_pix2pix.ipynb
├── diffusion.ipynb
├── transformer.ipynb
├── VAE_ForensicSketchToRealisticImage.ipynb
├── Frontend.ipynb
└── README.md
```

## Limitations

- Small paired dataset with synthetic (edge-detected) sketches; real hand-drawn forensic sketches vary widely and may generalize poorly.
- Identity preservation is not guaranteed when sketches are ambiguous or incomplete.
- Training constrained to a single GPU; GAN and diffusion training are sensitive to hyperparameters.
- Possible demographic bias inherited from the training data.

## Future Scope

Hybrid diffusion + transformer models, facial landmark or attribute supervision, text + sketch input, testing on real forensic sketches, lightweight models for fast inference, real-time interactive UI, and explainability (attention maps).

## Ethical Considerations

Generated faces must not be treated as proof of identity. Face data is biometric/personal data: use only datasets collected with consent, evaluate dataset bias, and do not use outputs for impersonation, identity fraud or misinformation. This project is for academic research and prototyping.

## Tech Stack

Python, PyTorch, Hugging Face Diffusers, Albumentations, TorchMetrics, LPIPS, CLIP, Gradio, Google Colab.

## License

Add a license of your choice (e.g. MIT).
