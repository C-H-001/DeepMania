<div align="center">

# DeepMania
![Status](https://img.shields.io/badge/Status-Work_in_Progress-orange) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

**[English](README.md) | [简体中文](README_zh.md)**

> ⚠️ **Project Disclaimer (WIP)**
> 1. **Note Type**: The model currently generates **Rice (single notes) only**. Long Note (LN) support is under development.
> 2. **Difficulty Accuracy**: The generated map's actual Star Rating (SR) may deviate slightly from the target. The script will calculate the real SR and rename the file automatically upon completion.
> 3. **Timing**: To ensure a playable experience, you must provide accurate **Timing Points** (BPM & Offset).
>
</div>

# DeepMania: Diffusion-Based Osu!Mania 4K Generator

DeepMania is an advanced framework for automatically generating Osu!Mania 4K beatmaps. Unlike traditional rule-based generators, DeepMania utilizes a **Conditional Diffusion Model** trained on thousands of Ranked maps. It learns to generate patterns that feel "human-made" based on audio features and a target difficulty (Star Rating).

<img width="1500" height="1200" alt="image" src="https://github.com/user-attachments/assets/ddc6a755-0c5f-4d9f-8c23-db6cab1df38c" />
<div align="center">(Cyber Inductance -- IcyWorld)</div>

## ✨ Key Features

*   **81-Channel Explicit Feature Injection**: Combines **Mel Spectrogram** (Melody/Timbre, 80ch) and **Onset Envelope** (Rhythm/Impact, 1ch) as inputs. The model doesn't just "guess" the beat; it explicitly "sees" the impacts, resulting in high timing precision.
*   **Gaussian Heatmap Prediction**: Instead of a binary grid, the model predicts a Gaussian-blurred probability map. This solves the "mode collapse" issue common in sparse data training and allows for fine-tuned peak picking.
*   **Smart Beat-wise Competition Quantization**: 
    *   Raw AI output is continuous. To solve the ambiguity between **Straight (1/4)** and **Swing (1/3)** rhythms, we use a competition algorithm.
    *   For each measure, the algorithm calculates the fitting error for both "Straight" and "Swing" grids.
    *   **Winner Takes All**: The rhythm with the lower error forces all notes in that measure to snap to its grid, ensuring local rhythmic consistency.
*   **Dynamic Complexity Limiting**: Low SR maps automatically block complex divisors (like 1/6 or 1/8 streams), ensuring the output matches the target player's skill level.

## 🛠️ Installation

Python 3.8+ and CUDA (recommended) are required.

```bash
# Clone the repository
git clone https://github.com/YourRepo/DeepMania.git
cd DeepMania

# Install dependencies
pip install -r requirements.txt
```

## 📥 Download Model

Pre-trained model weights (`best.pt`) are required.
Please download the latest weights from Hugging Face:

👉 **[Hugging Face Download Link](https://huggingface.co/AzusawaKohane/DeepMania/tree/main)**

**Setup:**
1. Download the `.pt` file.
2. Place it inside the `checkpoints/` directory in the project root.

## 🚀 Usage (Inference)

### Command Line Interface

```bash
python inference.py \
  --audio "songs/test_song.mp3" \
  --model "checkpoints/best.pt" \
  --timing "1020,333.33,4,2,1,60,1,0" \
  --sr 4.5 \
  --out "output.osu"
```

### Arguments

| Argument | Description |
| :--- | :--- |
| `--audio` | Path to the input MP3 file. |
| `--model` | Path to the model checkpoint (`.pt`). |
| `--timing` | **[Critical]** The Timing Point string (Red Line). Format: `Offset,BeatLength,...`. Copy this directly from the `[TimingPoints]` section of an existing `.osu` file. |
| `--sr` | Target Star Rating. Recommended range: **2.0 - 5.5**. |
| `--threshold` | (Optional) Note detection threshold. Default `0.55`. Lowering it creates denser maps; raising it makes them sparser. |
| `--out` | Output filename. **Note**: The script will automatically rename this file with the actual calculated SR (e.g., `output_Real4.52.osu`) after generation. |

## 🧠 Technical Pipeline

1.  **Feature Extraction**: Audio is converted to Mel Spectrogram + Onset Strength, concatenated, and normalized to `[-1, 1]`.
2.  **Diffusion**: A U-Net with Self-Attention receives noise and audio features, iteratively denoising it into a beatmap heatmap over 1000 steps, guided by the `Target SR`.
3.  **Peak Picking**: Raw timestamps are extracted from the heatmap.
4.  **Offset Correction**: The physical delay introduced during training (`TRAIN_SHIFT_FRAMES`, approx 70ms) is subtracted.
5.  **Competition Quantization**: 
    *   Notes are grouped by measure.
    *   Straight vs. Swing fitting errors are compared.
    *   Notes are snapped to the optimal grid and written to the `.osu` file.

## 📂 Project Structure

*   `inference.py`: Main inference script with post-processing and quantization logic.
*   `model.py`: 1D U-Net model definition with Self-Attention.
*   `dataset.py`: Data loader with real-time Gaussian blurring and feature injection.
*   `train.py`: Training loop using Cosine Schedule.
*   `sr_calculator.py`: Difficulty calculator ported from osu-tools.

## 🤝 Acknowledgements

Special thanks to the open-source community:

*   **[Star-Rating-Rebirth](https://github.com/sunnyxxy/Star-Rating-Rebirth)**: The core Strain System logic used for SR calculation and training labels is adapted from this repository.
