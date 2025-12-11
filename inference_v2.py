import argparse
import torch
import numpy as np
import librosa
import os
import math
import scipy.signal
from tqdm import tqdm
from model_v2 import ManiaDiffuserV2 

CONFIG = {
    'SR': 22050, 'HOP_LENGTH': 512, 'N_FFT': 2048, 'N_MELS': 80,
    'KEYS': 4, 'DEVICE': "cuda" if torch.cuda.is_available() else "cpu",
    'COLUMN_X': [64, 192, 320, 448]
}
TRAIN_SHIFT_FRAMES = 2.8 

class DiffusionSampler:
    def __init__(self, model, checkpoint_path, timesteps=1000):
        self.model = model
        self.timesteps = timesteps
        self.device = CONFIG['DEVICE']
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        self.beta = self.cosine_beta_schedule(timesteps).to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.alpha_hat_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alpha_hat[:-1]])

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    @torch.no_grad()
    def sample(self, audio_input, target_sr, ln_ratio_val):
        b, c, l = audio_input.shape
        # 生成 8 通道 (4 Head + 4 Body)
        img = torch.randn((b, 8, l), device=self.device)
        sr_tensor = torch.tensor([target_sr], dtype=torch.float32).to(self.device)
        ln_tensor = torch.tensor([ln_ratio_val], dtype=torch.float32).to(self.device)
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc="Sampling", total=self.timesteps):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            # 传入 ln_tensor
            predicted_noise = self.model(img, audio_input, t, sr_tensor, ln_tensor)
            
            alpha = self.alpha[t][:, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None]
            beta = self.beta[t][:, None, None]
            if i > 0: noise = torch.randn_like(img)
            else: noise = torch.zeros_like(img)
            img = (1 / torch.sqrt(alpha)) * (img - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            
        img = (img.clamp(-1, 1) + 1) / 2
        return img.cpu().numpy()

def prepare_audio_83ch(audio_path):
    y, sr = librosa.load(audio_path, sr=CONFIG['SR'])
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=CONFIG['N_FFT'], hop_length=CONFIG['HOP_LENGTH'], n_mels=80)
    log_mel = librosa.power_to_db(melspec, ref=np.max) # [80, T]
    
    # 3 Band Onset
    def get_onset(y, fmin, fmax):
        # 注意: 这里用 librosa 直接算，更简单
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=CONFIG['HOP_LENGTH'], fmin=fmin, fmax=fmax)
        if onset.max() > 0: onset = onset/onset.max()*2.0 - 1.0
        else: onset = np.zeros_like(onset) - 1.0
        return torch.tensor(onset).float().unsqueeze(0).unsqueeze(0)

    # 简单起见，这里演示用音频分频计算 (比Mel分频略慢但更准)，也可以复用 dataset.py 的 Mel 分切逻辑
    # 为了保持和 dataset.py 100% 一致，建议复用 dataset.py 的 _calc_onset 逻辑
    # 这里简略写 Mel 切分版:
    t_low = librosa.onset.onset_strength(S=log_mel[0:15, :], sr=sr)
    t_mid = librosa.onset.onset_strength(S=log_mel[15:50, :], sr=sr)
    t_high = librosa.onset.onset_strength(S=log_mel[50:, :], sr=sr)
    
    def norm(o):
        if o.max()>1e-4: return o/o.max()*2.0-1.0
        return np.zeros_like(o)-1.0
    
    t_low = torch.tensor(norm(t_low)).float().unsqueeze(0).unsqueeze(0)
    t_mid = torch.tensor(norm(t_mid)).float().unsqueeze(0).unsqueeze(0)
    t_high = torch.tensor(norm(t_high)).float().unsqueeze(0).unsqueeze(0)
    
    mel_tensor = torch.tensor(log_mel/40.0+1.0).float().unsqueeze(0)
    combined = torch.cat([mel_tensor, t_low, t_mid, t_high], dim=1).to(CONFIG['DEVICE'])
    
    length = combined.shape[2]
    pad_len = math.ceil(length/32)*32 - length
    if pad_len > 0: combined = torch.nn.functional.pad(combined, (0, pad_len))
    return combined, length

# ... (quantize 逻辑代码量较大，建议保留原版并增加 end_time 吸附，参考之前的 quantize_measure_wise_ln) ...
# 为了完整性，这里给出一个简单的 HitObject 生成逻辑

def grid_to_hitobjects(grid, beat_len, offset, threshold=0.5):
    raw_notes = []
    frame_ms = CONFIG['HOP_LENGTH'] / CONFIG['SR'] * 1000
    offset_correction = TRAIN_SHIFT_FRAMES * frame_ms
    
    grid_heads = grid[:4, :]
    grid_bodies = grid[4:, :] # -1 ~ 1
    T = grid.shape[1]
    
    for k in range(4):
        peaks, _ = scipy.signal.find_peaks(grid_heads[k], height=threshold, distance=2)
        for p in peaks:
            start_time = p * frame_ms - offset_correction
            
            # Check Body
            # 往后看3帧的平均值
            body_val = np.mean(grid_bodies[k, p : min(p+3, T)])
            is_ln = False
            end_time = -1
            
            # Body > -0.2 (接近1)
            if body_val > -0.2:
                # 找下降沿
                for f in range(p+1, T):
                    if grid_bodies[k, f] < -0.5:
                        is_ln = True
                        end_time = f * frame_ms - offset_correction
                        break
                        
            if is_ln and (end_time - start_time) < 30: is_ln = False
            
            raw_notes.append((start_time, k, is_ln, end_time))
            
    # 这里省略复杂的 Quantize，直接写简单的
    hit_objects = []
    for t, k, ln, t_end in raw_notes:
        x = CONFIG['COLUMN_X'][k]
        if ln:
            hit_objects.append(f"{x},192,{int(t)},128,0,{int(t_end)}:0:0:0:0:")
        else:
            hit_objects.append(f"{x},192,{int(t)},1,0,0:0:0:0:")
    return hit_objects

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timing", required=True)
    parser.add_argument("--sr", type=float, default=3.5)
    parser.add_argument("--ln_ratio", type=float, default=-1.0, help="-1=Auto, 0-1=Manual")
    parser.add_argument("--out", default="output.osu")
    args = parser.parse_args()
    
    model = ManiaDiffuserV2(in_channels=8, audio_channels=83).to(CONFIG['DEVICE'])
    sampler = DiffusionSampler(model, args.model)
    
    audio, length = prepare_audio_83ch(args.audio)
    
    print(f"Generating SR={args.sr}, LN={args.ln_ratio}...")
    grid = sampler.sample(audio, args.sr, args.ln_ratio)[0]
    grid = grid[:, :length]
    
    # 简单的 Timing 解析
    offset, beat_len = float(args.timing.split(',')[0]), float(args.timing.split(',')[1])
    
    objs = grid_to_hitobjects(grid, beat_len, offset)
    
    with open(args.out, "w") as f:
        f.write("osu file format v14\n[General]\nMode: 3\n[HitObjects]\n")
        for line in objs: f.write(line+"\n")
    print("Done.")