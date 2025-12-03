(Based on Gemini3 Pro)
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

# DeepMania: Diffusion-Based Osu!Mania 4K Generator (v1.0)

DeepMania 是一个基于 **Conditional Diffusion Model (条件扩散模型)** 的 Osu!Mania 4K 谱面自动生成器。

DeepMania 学习了数千张 Ranked 谱面的分布，能够根据音频Mels谱特征和目标星数（Star Rating）生成具有“人手感”的谱面。

<img width="2700" height="1500" alt="image" src="https://github.com/user-attachments/assets/fbe76104-53ba-4dbf-a3e7-a560c8a25d91" />

## ✨ 核心特性

*   **多模态输入 (Multi-modal Input)**: 结合了 **Mel Spectrogram** (旋律/音色) 和 **Onset Envelope** (重音/节奏) 双通道输入，确保生成的 Note 紧贴音乐重音。
*   **难度可控 (Difficulty Conditioning)**: 支持指定目标星数 (SR)，模型会根据 SR 调整生成的密度和 Pattern 复杂度（如低星出单点，高星出纵连/交互）。
*   **高斯热力图预测 (Gaussian Heatmap)**: 摒弃了传统的二值化 Grid，使用高斯模糊后的概率图进行训练，解决了稀疏数据下的模式崩塌问题。
*   **智能节奏量化 (Beat-wise Competition Quantization)**: 
    *   引入了基于小节的节奏竞争算法。
    *   模型会自动判断当前一拍是 **Straight (1/2, 1/4, 1/8)** 还是 **Swing (1/3, 1/6)**。
    *   保证了局部节奏的一致性，消除了“杂乱无章”的 AI 生成感。

## 🛠️ 安装

需要 Python 3.8+ 和 CUDA 环境（推荐）。

```bash
pip install -r requirements.txt
```

## 📥 模型下载

本项目需要预训练的模型权重（`best.pt`）才能运行。
请前往 Hugging Face 下载最新权重的模型：

👉 **[Hugging Face 下载链接](https://huggingface.co/AzusawaKohane/DeepMania/tree/main)**

**配置步骤:**
1. 点击上方链接下载 `.pt` 模型文件 (推荐下载 `best.pt`)。
2. 在项目根目录下新建一个名为 `checkpoints` 的文件夹。
3. 将下载好的文件放入 `checkpoints/` 目录中。

```bash
mkdir checkpoints
# 将下载的 best.pt 放入此文件夹

## 🚀 使用方法 (Inference)

你需要准备：
1.  一个音频文件 (`.mp3`)。
2.  该音频的 Timing Point 数据 (从原谱的 `.osu` 文件中复制，或者自己测定)(目前该模型对于BPM和Offset的预测还不准确)。
3.  预训练好的模型权重 (`.pt`)。

### 命令行示例

```bash
python inference.py \
  --audio "songs/freedom_dive.mp3" \
  --model "checkpoints/best.pt" \
  --timing "243,266.666,4,2,1,60,1,0" \
  --sr 5.5 \
  --out "output_freedom_dive.osu"
```

### 参数说明

*   `--audio`: MP3 文件路径。
*   `--model`: 模型权重路径。
*   `--timing`: Osu 格式的 Timing Point 字符串（红线）。格式为 `Offset,BeatLength,...`。这对于保证生成的谱面与音乐对齐至关重要。
*   `--sr`: 目标星数。建议范围 2.0 - 6.0。
*   `--threshold`: (可选) Note 判定阈值，默认 0.5。调低会增加 Note 数量，调高会更稀疏。

## 🧠 技术原理细节

### 1. 训练数据的构建
我们将谱面转换为 `[Time, 4]` 的 Grid，并对时间轴应用高斯模糊（Sigma=1.0）。音频部分提取 Mel 谱图，并额外计算 Onset Strength 作为一个显式的 Channel 注入模型，使模型能够轻松捕捉到音符的位置。

### 2. 模型架构
使用修改版的 1D U-Net。
*   **Input**: Noisy Grid (4ch) + Mel (80ch) + Onset (1ch) = 85 Channels。
*   **Conditioning**: Time Embedding + Star Rating Embedding (通过 FiLM 层注入)。

### 3. 后处理：小节竞争算法 (Beat-wise Competition)
AI 生成的时间是连续的浮点数。为了量化到 Osu 的整数网格，我们不简单地吸附到最近点。
算法会分析每一拍内的所有 Note：
*   假设它是直拍 (Straight)，计算总误差。
*   假设它是三连音 (Swing)，计算总误差。
*   **Winner Takes All**: 哪种假设误差更小，这一拍的所有 Note 就强制吸附到该节奏系下。这避免了 1/4 和 1/3 混杂出现的“鬼畜”节奏。

### 4. 动态节奏复杂度 (Dynamic Rhythm Complexity)
为了保证谱面的合理性，后处理程序会根据用户输入的 `target_sr` (目标星数) 动态解锁允许的节拍细分。这意味着**低星图不会出现不合理的高速连打**。

当前的阈值逻辑如下：

| 节奏类型 | 基础节拍 | 中等难度解锁 | 高难度解锁 |
| :--- | :--- | :--- | :--- |
| **Straight (直拍)** | 1/1, 1/2, 1/4 | **SR > 4.5**: 允许 1/8 (Stream) | **SR > 6.0**: 允许 1/16 (Dump/Tech) |
| **Swing (摇摆/三连)** | 1/1, 1/3 | **SR > 4.0**: 允许 1/6 (Fast Triplet) | **SR > 5.5**: 允许 1/12 (Hyper-fast Triplet) |

*例如：如果您生成 SR=3.5 的谱面，算法将自动屏蔽 1/6, 1/8, 1/12 等复杂节奏，强制吸附到最简单的节拍上，以保证新手可玩性。*

## 📂 项目结构

*   `inference.py`: 推理主程序，包含完整的后处理逻辑。
*   `model.py`: PyTorch U-Net 模型定义。
*   `dataset.py`: 数据加载与实时高斯模糊处理。
*   `train.py`: 训练循环代码。
*   `sr_calculator.py`: (辅助) 用于计算谱面难度的逻辑。

## 📜 License
MIT

## 🤝 致谢 (Acknowledgements)

特别感谢开源社区的贡献：

*   **[Star-Rating-Rebirth](https://github.com/sunnyxxy/Star-Rating-Rebirth)** (作者 **[sunnyxxy](https://github.com/sunnyxxy)**):
    本项目中的 `osu_file_parser.py` 和 `sr_calculator.py` 引用并修改自该仓库。我们使用其精确的 Strain 系统算法来为扩散模型计算训练数据的难度标签。

### 还有GEMINI3 Pro!!!!
