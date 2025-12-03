<div align="center">

# DeepMania
![Status](https://img.shields.io/badge/Status-Work_in_Progress-orange)
**[English](README.md) | [简体中文](README_zh.md)**
> This project is currently in **active development**.
> 1. **Note Type**: The model currently supports **Rice (single notes) only**. Support for Long Notes (LNs) is under development.
> 2. **Difficulty Control**: The actual Star Rating (SR) of the generated map may deviate slightly from the target value.
> 3. **Timing**: To ensure the best gameplay experience, please provide accurate **Timing Points** (BPM & Offset). Automatic detection is currently experimental and used for reference only.
>
> ---
>
> 本项目仍处于**开发阶段**。
> 1. **Note 类型**: 目前模型仅生成 **Rice (单点)**，面条 (Long Notes) 功能开发中。
> 2. **难度控制**: 输出的实际星数 (SR) 可能会在目标值上下浮动。
> 3. **Timing**: 为了保证游玩体验，请务必提供准确的 **Timing Points** (BPM & Offset)，自动检测目前仅作辅助。
</div>

---
# DeepMania: Diffusion-Based Osu!Mania 4K Generator (v1.0)

DeepMania is an automatic Osu!Mania 4K beatmap generator based on a **Conditional Diffusion Model**.

DeepMania has learned the distribution of thousands of Ranked beatmaps and can generate charts with a "human touch" based on audio Mel-spectrogram features and a target Star Rating (SR).

<img width="2700" height="1500" alt="image" src="https://github.com/user-attachments/assets/e98a29ed-aa0f-4084-9e58-c5a26862abb4" />


## ✨ Key Features

*   **Multi-modal Input**: Combines **Mel Spectrogram** (melody/timbre) and **Onset Envelope** (rhythm/accents) as dual-channel inputs, ensuring generated notes align closely with musical accents.
*   **Difficulty Conditioning**: Supports specifying a target Star Rating (SR). The model adjusts note density and pattern complexity (e.g., single notes for low SR, jump-trills/streams for high SR) based on the input SR.
*   **Gaussian Heatmap**: Abandons traditional binary grids in favor of Gaussian-blurred probability maps during training, solving mode collapse issues often found in sparse data generation.
*   **Smart Rhythm Quantization (Beat-wise Competition)**: 
    *   Introduces a beat-wise rhythm competition algorithm.
    *   The model automatically determines if the current beat follows a **Straight** rhythm (1/2, 1/4, 1/8) or a **Swing** rhythm (1/3, 1/6).
    *   This ensures local rhythmic consistency and eliminates the "chaotic" feel often seen in AI generation.

## 🛠️ Installation

Requires Python 3.8+ and a CUDA environment (Recommended).

```bash
pip install -r requirements.txt
```
## 📥 Model Weights

To generate beatmaps, you need the pre-trained model weights (`best.pt`).
You can download them from our Hugging Face repository:

👉 **[Download on Hugging Face](https://huggingface.co/AzusawaKohane/DeepMania/tree/main)**

**Setup:**
1. Download the `.pt` file (e.g., `best.pt`).
2. Create a folder named `checkpoints` in the root directory.
3. Place the downloaded file into `checkpoints/`.

```bash
mkdir checkpoints
# Move your downloaded best.pt here


## 🚀 Usage (Inference)

You need to prepare:
1.  An audio file (`.mp3`).
2.  Timing Point data for the audio (Copy this from the original `.osu` file or measure it yourself; the model currently does not accurately predict BPM/Offset).
3.  Pre-trained model weights (`.pt`).

### CLI Example

```bash
python inference.py \
  --audio "songs/freedom_dive.mp3" \
  --model "checkpoints/best.pt" \
  --timing "243,266.666,4,2,1,60,1,0" \
  --sr 5.5 \
  --out "output_freedom_dive.osu"
```

### Parameters

*   `--audio`: Path to the input MP3 file.
*   `--model`: Path to the model checkpoint.
*   `--timing`: Osu-formatted Timing Point string (Red Line). Format: `Offset,BeatLength,...`. This is crucial for aligning the generated chart with the music.
*   `--sr`: Target Star Rating. Recommended range: 2.0 - 6.0.
*   `--threshold`: (Optional) Note detection threshold, default is 0.5. Lower values increase note count; higher values make it sparser.

## 🧠 Technical Details

### 1. Data Construction
Beatmaps are converted into `[Time, 4]` grids with Gaussian blur applied along the time axis (Sigma=1.0). For audio, we extract Mel Spectrograms and additionally calculate **Onset Strength** as an explicit channel injection, allowing the model to easily capture note placements.

### 2. Model Architecture
Uses a modified 1D U-Net.
*   **Input**: Noisy Grid (4ch) + Mel (80ch) + Onset (1ch) = 85 Channels.
*   **Conditioning**: Time Embedding + Star Rating Embedding (injected via FiLM layers).

### 3. Post-processing: Beat-wise Competition Algorithm
The model generates continuous floating-point timestamps. To snap these to the Osu integer grid, we avoid simple nearest-neighbor snapping.
The algorithm analyzes all notes within a single beat:
*   Hypothesis A: It is a **Straight** beat (1/4, 1/8). Calculate total error.
*   Hypothesis B: It is a **Swing** beat (1/3, 1/6). Calculate total error.
*   **Winner Takes All**: All notes in that beat are forced to snap to the grid of the winning hypothesis. This prevents the chaotic mixture of 1/4 and 1/3 rhythms.

### 4. Dynamic Rhythm Complexity
To ensure playability and structural logic, the post-processing algorithm dynamically unlocks allowed rhythmic divisors based on the provided `target_sr` (Target Star Rating). This ensures that **low-star maps will not contain unreasonable high-speed streams.**

The current threshold logic is as follows:

| Rhythm Type | Base Divisors | Intermediate Unlock | Expert Unlock |
| :--- | :--- | :--- | :--- |
| **Straight** | 1/1, 1/2, 1/4 | **SR > 4.5**: Adds 1/8 (Stream) | **SR > 6.0**: Adds 1/16 (Dump/Tech) |
| **Swing** | 1/1, 1/3 | **SR > 4.0**: Adds 1/6 (Fast Triplet) | **SR > 5.5**: Adds 1/12 (Hyper-fast Triplet) |

*Example: If you generate a map with SR=3.5, the algorithm automatically disables complex rhythms like 1/6, 1/8, and 1/12, forcing notes to snap to simpler grids to ensure accessibility for beginners.*

## 📂 Project Structure

*   `inference.py`: Main inference script including the full post-processing logic.
*   `model.py`: PyTorch U-Net model definition.
*   `dataset.py`: Data loading and real-time Gaussian blur processing.
*   `train.py`: Training loop code.
*   `sr_calculator.py`: (Helper) Logic for calculating beatmap difficulty.

## 📜 License

MIT

## 🤝 Acknowledgements

Special thanks to the open-source community for their contributions:

*   **[Star-Rating-Rebirth](https://github.com/sunnyxxy/Star-Rating-Rebirth)** by **[sunnyxxy](https://github.com/sunnyxxy)**:
    The files `osu_file_parser.py` and `sr_calculator.py` in this project are forked and adapted from this repository. We rely on its accurate Strain System implementation to generate Star Rating labels for training our diffusion model.

### AND GEMINI3 PRO!!!!
