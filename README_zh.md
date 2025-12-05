<div align="center">

# DeepMania
![Status](https://img.shields.io/badge/Status-Work_in_Progress-orange) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

**[English](README.md) | [简体中文](README_zh.md)**

> ⚠️ **项目注意事项 (WIP)**
> 1. **Note 类型**: 目前模型仅生成 **Rice (单点)**，面条 (Long Notes) 功能开发中。
> 2. **难度控制**: 输出的实际星数 (SR) 可能会在目标值上下浮动。脚本会在生成结束后计算实际 SR 并自动重命名文件。
> 3. **Timing**: 为了保证游玩体验，请务必提供准确的 **Timing Points** (BPM & Offset)。
>
</div>

# DeepMania: 基于扩散模型的 Osu!Mania 4K 谱面生成器

DeepMania 是一个先进的 Osu!Mania 4K 谱面自动生成框架。不同于传统的规则生成器，它基于 **Conditional Diffusion Model (条件扩散模型)**，学习了数千张 Ranked 谱面的分布规律，能够根据音频特征和目标难度生成具有“人手感”的谱面。

<img src="https://github.com/user-attachments/assets/fbe76104-53ba-4dbf-a3e7-a560c8a25d91" alt="DeepMania Demo" width="100%">

## ✨ 核心特性

*   **81通道显式特征注入**: 结合了 **Mel Spectrogram** (旋律/音色, 80ch) 和 **Onset Envelope** (重音/节奏, 1ch) 作为输入。模型不再需要“猜”哪里有鼓点，而是直接“看”到重音，实现了极高的时间对齐精度。
*   **高斯热力图预测 (Gaussian Heatmap)**: 摒弃了传统的二值化 Grid，使用高斯模糊后的概率图进行训练。这解决了稀疏数据下的模式崩塌问题，并允许通过 Peak Picking 算法微调手感。
*   **智能小节竞争量化 (Beat-wise Competition Quantization)**: 
    *   这是 DeepMania 的杀手锏。AI 生成的时间通常是连续的。
    *   为了解决 **1/4 (直线)** 和 **1/3 (摇摆/三连音)** 的混淆问题，我们引入了基于小节的竞争算法。
    *   程序会计算每一小节在“直拍”和“摇摆拍”下的拟合误差，**Winner Takes All**，确保了节奏的局部一致性。
*   **动态难度限制**: 低星图自动屏蔽 1/6, 1/8 等高难节奏，确保生成的谱面符合目标星数的玩家水平。

## 🛠️ 安装

需要 Python 3.8+ 和 CUDA 环境（推荐）。

```bash
# 克隆仓库
git clone https://github.com/YourRepo/DeepMania.git
cd DeepMania

# 安装依赖
pip install -r requirements.txt
```

## 📥 模型下载

本项目需要预训练的模型权重（`best.pt`）才能运行。
请前往 Hugging Face 下载最新权重的模型：

👉 **[Hugging Face 下载链接 (示例)](https://huggingface.co/AzusawaKohane/DeepMania/tree/main)**

**配置步骤:**
1. 下载 `.pt` 模型文件。
2. 放入项目根目录下的 `checkpoints/` 文件夹中。

## 🚀 使用方法 (Inference)

### 命令行参数

```bash
python inference.py \
  --audio "songs/test_song.mp3" \
  --model "checkpoints/best.pt" \
  --timing "1020,333.33,4,2,1,60,1,0" \
  --sr 4.5 \
  --out "output.osu"
```

### 参数说明

| 参数 | 说明 |
| :--- | :--- |
| `--audio` | 输入的 MP3 音频文件路径。 |
| `--model` | 预训练模型权重路径 (`.pt`)。 |
| `--timing` | **[关键]** 也就是红线。格式为 `Offset,BeatLength,...`。请直接从原谱 `.osu` 文件的 `[TimingPoints]` 处复制第一行。 |
| `--sr` | 目标星数 (Star Rating)。推荐范围 **2.0 - 5.5**。 |
| `--threshold` | (可选) Note 判定阈值，默认 `0.55`。调低(如0.4)会增加 Note 密度，调高会更稀疏。 |
| `--out` | 输出文件名。**注意**：脚本运行结束后，会根据计算出的实际 SR 自动重命名该文件（例如 `output_Real4.52.osu`）。 |

## 🧠 技术原理 (Pipeline)

1.  **特征提取**: 音频被转换为 Mel 谱图，同时计算 Onset Strength。两者拼接并归一化到 `[-1, 1]`。
2.  **扩散生成**: U-Net 接收噪声和音频特征，在 `Target SR` 的引导下，通过 1000 步逆向扩散生成谱面热力图。
3.  **寻峰 (Peak Picking)**: 从热力图中提取 Note 的原始时间点，此时带有物理延迟。
4.  **偏移修正**: 自动扣除训练时引入的 `TRAIN_SHIFT_FRAMES` (约70ms) 物理延迟。
5.  **竞争量化**: 
    *   将 Note 按小节分组。
    *   比较 Straight (1/2, 1/4) 和 Swing (1/3, 1/6) 的拟合度。
    *   强制吸附到最优网格，并写入 `.osu` 文件。

## 📂 项目结构

*   `inference.py`: 推理主程序，包含完整的后处理与量化逻辑。
*   `model.py`: 包含 Self-Attention 的 1D U-Net 模型定义。
*   `dataset.py`: 数据加载器，包含实时高斯模糊与特征注入。
*   `train.py`: 训练循环代码，使用 Cosine Schedule。
*   `sr_calculator.py`: 移植自 osu-tools 的难度计算器。

## 🤝 致谢

特别感谢开源社区的贡献：

*   **[Star-Rating-Rebirth](https://github.com/sunnyxxy/Star-Rating-Rebirth)**: 本项目的 SR 计算核心逻辑引用自该仓库。


