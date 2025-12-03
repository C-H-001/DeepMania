import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os
import scipy.ndimage # 需要用到这个库做模糊
import librosa

class ManiaDataset(Dataset):
    def __init__(self, data_dir, sigma=1.0):
        """
        sigma: 高斯模糊的标准差。
               你的 HOP_LENGTH=512 (约23ms)。
               sigma=1.0 意味着模糊半径约 23ms，这对于捕捉节奏非常合适。
        """
        self.file_list = glob.glob(os.path.join(data_dir, "*.npz"))
        self.data_cache = []
        self.sigma = sigma

        print(f"正在加载数据集: {len(self.file_list)} 个文件...")

        # 预加载数据索引
        for f_path in self.file_list:
            try:
                # 使用 mmap_mode='r' 可以极大节省内存，不需要一次性把 npz 读入 RAM
                # 但如果你内存够大 (几十GB)，可以直接 load 进去加速
                with np.load(f_path) as data:
                    mels = data['mels']     # [N, Time, 80]
                    charts = data['charts'] # [N, Time, 4] -> int8 (0 or 1)
                    srs = data['local_srs'] # [N]

                    for i in range(len(mels)):
                        # 这里我们只存 numpy 数组，转换逻辑放在 __getitem__
                        self.data_cache.append({
                            'mel': mels[i],
                            'chart': charts[i],
                            'sr': srs[i]
                        })
            except Exception as e:
                print(f"加载错误 {f_path}: {e}")

        print(f"总切片数: {len(self.data_cache)}")

    def __len__(self):
        return len(self.data_cache)

    # 修改 dataset.py 中的 __getitem__ 方法

    def __getitem__(self, idx):
        item = self.data_cache[idx]

        # --- 1. Audio 处理 ---
        # 原始存储的是 [Time, 80] 的 Log-Mel
        mel_np = item['mel']  # numpy array

        # 计算 Onset Strength (关键步骤)
        # librosa 需要 [n_mels, Time] 格式
        # S 参数接受 Log-Mel (dB)
        onset_env = librosa.onset.onset_strength(S=mel_np.T, sr=22050)

        # 归一化 Onset 到 [-1, 1] 以匹配 Diffusion 输入分布
        if onset_env.max() > 0:
            onset_env = onset_env / onset_env.max()  # [0, 1]
            onset_env = onset_env * 2.0 - 1.0  # [-1, 1]
        else:
            onset_env = np.zeros_like(onset_env) - 1.0  # 如果没声音，就是 -1

        # 转为 Tensor
        # Mel: [80, Time]
        mel_tensor = torch.tensor(mel_np, dtype=torch.float32).transpose(0, 1)
        # 归一化 Mel (假设 -80dB ~ 0dB) -> [-1, 1]
        mel_tensor = mel_tensor / 40.0 + 1.0

        # Onset: [1, Time]
        onset_tensor = torch.tensor(onset_env, dtype=torch.float32).unsqueeze(0)

        # === 融合 ===
        # 结果 shape: [81, Time]
        audio_feature = torch.cat([mel_tensor, onset_tensor], dim=0)

        # --- 2. Chart 处理 (保持高斯模糊逻辑) ---
        raw_chart = item['chart'].astype(np.float32)
        heatmap = np.zeros_like(raw_chart)

        for k in range(raw_chart.shape[1]):
            if raw_chart[:, k].max() > 0:
                blurred = scipy.ndimage.gaussian_filter1d(raw_chart[:, k], sigma=self.sigma)
                peak = blurred.max()
                if peak > 0:
                    heatmap[:, k] = blurred / peak

        chart_tensor = torch.tensor(heatmap, dtype=torch.float32).transpose(0, 1)
        chart_tensor = chart_tensor * 2.0 - 1.0

        sr = torch.tensor(item['sr'], dtype=torch.float32).unsqueeze(0)

        # 返回融合后的 audio_feature
        return audio_feature, chart_tensor, sr