import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os
import scipy.ndimage
import librosa

class ManiaDataset(Dataset):
    def __init__(self, data_dir, sigma=1.0):
        self.sigma = sigma
        self.index = [] 
        
        # 【改动】Lazy Loading 索引构建
        print("Indexing dataset...")
        for f_path in glob.glob(os.path.join(data_dir, "*.npz")):
            try:
                # mmap_mode='r' 只读头信息，不加载数据
                with np.load(f_path, mmap_mode='r') as data:
                    num_samples = data['local_srs'].shape[0]
                    for i in range(num_samples):
                        self.index.append((f_path, i))
            except Exception as e:
                print(f"Error indexing {f_path}: {e}")
        print(f"Indexed {len(self.index)} samples.")

    def __len__(self):
        return len(self.index)

    def _calc_onset(self, mel_db):
        onset = librosa.onset.onset_strength(S=mel_db, sr=22050)
        if onset.max() > 1e-4:
            onset = onset / onset.max() * 2.0 - 1.0
        else:
            onset = np.zeros_like(onset) - 1.0
        return onset

    def __getitem__(self, idx):
        f_path, inner_idx = self.index[idx]
        
        # 运行时读取
        with np.load(f_path, mmap_mode='r') as data:
            mel_np = data['mels'][inner_idx]     # [Time, 80]
            chart_np = data['charts'][inner_idx] # [Time, 8]
            sr_val = data['local_srs'][inner_idx]
            ln_ratio_val = data['ln_ratios'][inner_idx]

        # --- 1. Audio Feature Engineering (83ch) ---
        mel_db_T = mel_np.T # [80, Time]
        
        # 分频段 Onset
        onset_low = self._calc_onset(mel_db_T[0:15, :])
        onset_mid = self._calc_onset(mel_db_T[15:50, :])
        onset_high = self._calc_onset(mel_db_T[50:, :])
        
        t_low = torch.tensor(onset_low, dtype=torch.float32).unsqueeze(0)
        t_mid = torch.tensor(onset_mid, dtype=torch.float32).unsqueeze(0)
        t_high = torch.tensor(onset_high, dtype=torch.float32).unsqueeze(0)
        
        mel_tensor = torch.tensor(mel_np, dtype=torch.float32).transpose(0, 1)
        mel_tensor = mel_tensor / 40.0 + 1.0 # Norm
        
        audio_feature = torch.cat([mel_tensor, t_low, t_mid, t_high], dim=0) # [83, T]

        # --- 2. Chart Processing (Head+Body) ---
        raw_chart = chart_np.astype(np.float32).copy() # copy needed for mmap
        
        # Shift Compensation
        SHIFT_FRAMES = 3
        shifted = np.zeros_like(raw_chart)
        shifted[SHIFT_FRAMES:, :] = raw_chart[:-SHIFT_FRAMES, :]
        raw_chart = shifted
        
        # Split Head/Body
        heads = raw_chart[:, :4]
        bodies = raw_chart[:, 4:]
        
        # Head: Gaussian Blur
        heatmap_heads = np.zeros_like(heads)
        for k in range(4):
            if heads[:, k].max() > 0:
                blurred = scipy.ndimage.gaussian_filter1d(heads[:, k], sigma=self.sigma)
                peak = blurred.max()
                if peak > 0: heatmap_heads[:, k] = blurred / peak
        heatmap_heads = heatmap_heads * 2.0 - 1.0
        
        # Body: Linear [-1, 1]
        heatmap_bodies = bodies * 2.0 - 1.0
        
        final_chart = np.concatenate([heatmap_heads, heatmap_bodies], axis=1)
        chart_tensor = torch.tensor(final_chart, dtype=torch.float32).transpose(0, 1) # [8, T]

        # --- 3. Conditions ---
        sr = torch.tensor([sr_val], dtype=torch.float32)
        ln_ratio = torch.tensor([ln_ratio_val], dtype=torch.float32)

        return audio_feature, chart_tensor, sr, ln_ratio