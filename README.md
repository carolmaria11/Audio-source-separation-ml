# 🎧 Audio Source Separation

> Separate music into **Vocals · Drums · Bass · Other** using unsupervised models and Demucs.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org)

---

## What This Does

Two notebooks that take any WAV or MP3 file and split it into 4 stems:

| Notebook | Approach | Needs Labels? |
|---|---|---|
| `audio_separation.ipynb` | 6 unsupervised models (ICA → MixIT) | ❌ No |
| `audio_source_separation_demucs.ipynb` | Demucs (pretrained) | ✅ Pretrained weights |

---

## Models

### Unsupervised (no labels, no pretrained weights)

| Model | How It Works | SDR |
|---|---|---|
| **ICA** | Statistical de-mixing matrix | -0.92 dB |
| **NMF** | Spectrogram factorization `V ≈ W × H` | **5.79 dB** ✅ |
| **Autoencoder** | Shared encoder + 4 decoder heads | 1.37 dB |
| **VAE** | Probabilistic latent space + KL loss | 2.69 dB |
| **RVAE** | VAE + GRU for temporal context | 1.30 dB |
| **MixIT** | Self-supervised mixture-of-mixtures | 4.11 dB |

### Supervised

| Model | SDR |
|---|---|
| **Demucs (htdemucs)** | **15.14 dB** 🏆 |

> NMF wins among unsupervised models. Neural models underperform here due to limited training data (30 seconds). With more audio they would outperform NMF.

---
## Installation

```bash
# Clone
git clone https://github.com/yourusername/audio-source-separation.git
cd audio-source-separation

# Install
pip install librosa soundfile scikit-learn torch torchaudio numpy matplotlib scipy

# For Demucs only
pip install demucs
```

---

## Usage

### Unsupervised

```bash
jupyter notebook audio_separation.ipynb
```

1. Open **Cell 3** and set your audio file path:
```python
AUDIO_FILE_PATH = 'your_song.wav'
```
2. Run **Kernel → Restart & Run All**

> No audio file? Leave the path as-is — a synthetic test signal is generated automatically.

### Demucs

```bash
jupyter notebook audio_source_separation_demucs.ipynb
```

1. Open **Section 2** and set your audio file:
```python
INPUT_AUDIO_PATH = r"C:\Users\You\Music\song.wav"   # Windows
INPUT_AUDIO_PATH = "/home/user/music/song.wav"       # Mac/Linux
```
2. Run all cells — Demucs downloads model weights (~200MB) on first run

---

## SDR Guide

```
< 0 dB   →  Very Poor    (worse than silence)
0–3 dB   →  Poor         (heavy distortion)
3–6 dB   →  Marginal     (broad separation, artifacts present)
6–10 dB  →  Acceptable   (some distortion)
10–15 dB →  Good         (clearly intelligible)
> 15 dB  →  Excellent    (near-perfect)
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `librosa` | Audio loading and STFT |
| `scikit-learn` | ICA, NMF, KMeans |
| `torch` | Neural models (AE, VAE, RVAE, MixIT) |
| `soundfile` | Saving `.wav` files |
| `matplotlib` | Visualizations |
| `demucs` | Supervised separation (Demucs notebook only) |

---

<div align="center">
<sub>ICA · NMF · Autoencoder · VAE · RVAE · MixIT · Demucs</sub>
</div>
