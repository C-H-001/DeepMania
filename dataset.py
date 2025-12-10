import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os
import scipy.ndimage
import librosa

class ManiaDataset(Dataset):
    def __init__(self, data_dir, sigma=1.0):
        self.file_list = glob.glob(os.path.join(data_dir, "*.npz"))
        self.data_cache = [] # 建议后续参考我之前的建议改为 Lazy Loading，此处为了兼容你的原代码保持不变
        self.sigma = sigma

        print(f"正在加载数据集: {len(self.file_list)} 个文件...")

        for f_path in self.file_list:
            try:
                with np.load(f_path) as data:
                    mels = data['mels']     
                    charts = data['charts'] 
                    srs = data['local_srs'] 

                    for i in range(len(mels)):
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

    def _calc_onset(self, mel_db):
        """辅助函数：根据 Mel 谱 (dB) 计算归一化的 Onset Strength"""
        # librosa.onset.onset_strength 支持直接传入频谱 (S)
        # S 的形状要求是 [n_mels, Time]
        onset = librosa.onset.onset_strength(S=mel_db, sr=22050)
        
        # 归一化到 [-1, 1]
        if onset.max() > 1e-4:
            onset = onset / onset.max() # [0, 1]
            onset = onset * 2.0 - 1.0   # [-1, 1]
        else:
            onset = np.zeros_like(onset) - 1.0
        return onset

    def __getitem__(self, idx):
        item = self.data_cache[idx]

        # --- 1. Audio 处理 (关键修改区域) ---
        # 原始 mel_np shape: [Time, 80]
        mel_np = item['mel']
        
        # 转置为 [80, Time] 以供 librosa 使用
        mel_db_T = mel_np.T 

        # === 核心修改：分频段计算 Onset ===
        # 假设 80 个 Mel 频带。
        # 0-15: 低频 (Bass/Kick)
        # 15-50: 中频 (Snare/Vocals)
        # 50-80: 高频 (Hi-hats/Cymbals)
        
        onset_low = self._calc_onset(mel_db_T[0:15, :])
        onset_mid = self._calc_onset(mel_db_T[15:50, :])
        onset_high = self._calc_onset(mel_db_T[50:, :])
        
        # 转为 Tensor [1, Time]
        t_low = torch.tensor(onset_low, dtype=torch.float32).unsqueeze(0)
        t_mid = torch.tensor(onset_mid, dtype=torch.float32).unsqueeze(0)
        t_high = torch.tensor(onset_high, dtype=torch.float32).unsqueeze(0)

        # Mel 本身也要转 Tensor 并归一化 [80, Time]
        mel_tensor = torch.tensor(mel_np, dtype=torch.float32).transpose(0, 1)
        mel_tensor = mel_tensor / 40.0 + 1.0

        # === 融合 ===
        # 结果 shape: [80 + 3, Time] = [83, Time]
        audio_feature = torch.cat([mel_tensor, t_low, t_mid, t_high], dim=0)

        # --- 2. Chart 处理 (保持原样) ---
        raw_chart = item['chart'].astype(np.float32)

        # 训练时 Shift (如需修改请在这里改)
        SHIFT_FRAMES = 3 
        shifted_chart = np.zeros_like(raw_chart)
        shifted_chart[SHIFT_FRAMES:, :] = raw_chart[:-SHIFT_FRAMES, :]
        raw_chart = shifted_chart
            
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

        return audio_feature, chart_tensor, sr