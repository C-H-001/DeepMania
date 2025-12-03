import argparse
import torch
import numpy as np
import librosa
import os
import math
import scipy.signal
from tqdm import tqdm
from model import ManiaUNet

# ================= 配置 =================
CONFIG = {
    'SR': 22050,
    'HOP_LENGTH': 512,
    'N_FFT': 2048,
    'N_MELS': 80,
    'KEYS': 4,
    'DEVICE': "cuda" if torch.cuda.is_available() else "cpu",
    'COLUMN_X': [64, 192, 320, 448] 
}

# ================= 核心类与函数 =================

class DiffusionSampler:
    def __init__(self, model, checkpoint_path, timesteps=1000):
        self.model = model
        self.timesteps = timesteps
        self.device = CONFIG['DEVICE']
        
        print(f"Loading model from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.beta = torch.linspace(1e-4, 0.02, timesteps).to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.alpha_hat_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alpha_hat[:-1]])

    @torch.no_grad()
    def sample(self, audio_input, target_sr):
        b, c, l = audio_input.shape
        img = torch.randn((b, 4, l), device=self.device)
        sr_tensor = torch.tensor([target_sr], dtype=torch.float32).to(self.device)
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc="Sampling", total=self.timesteps):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            predicted_noise = self.model(img, audio_input, t, sr_tensor)
            
            alpha = self.alpha[t][:, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None]
            beta = self.beta[t][:, None, None]
            
            if i > 0:
                noise = torch.randn_like(img)
            else:
                noise = torch.zeros_like(img)
                
            img = (1 / torch.sqrt(alpha)) * (img - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            
        img = (img.clamp(-1, 1) + 1) / 2
        return img.cpu().numpy()

def parse_manual_timing(timing_str):
    parts = timing_str.split(',')
    offset = float(parts[0])
    beat_len = float(parts[1])
    bpm = 60000.0 / beat_len
    print(f"Manual Timing Parsed: Offset={offset}ms, BPM={bpm:.2f}")
    return offset, beat_len

def prepare_audio_with_onset(audio_path):
    y, sr = librosa.load(audio_path, sr=CONFIG['SR'])
    melspec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=CONFIG['N_FFT'], 
        hop_length=CONFIG['HOP_LENGTH'], n_mels=CONFIG['N_MELS']
    )
    log_mel = librosa.power_to_db(melspec, ref=np.max)
    onset_env = librosa.onset.onset_strength(S=log_mel, sr=sr)
    
    if onset_env.max() > 0:
        onset_env = onset_env / onset_env.max() * 2.0 - 1.0
    else:
        onset_env = np.zeros_like(onset_env) - 1.0
        
    norm_mel = log_mel / 40.0 + 1.0
    
    mel_tensor = torch.tensor(norm_mel, dtype=torch.float32).unsqueeze(0)
    onset_tensor = torch.tensor(onset_env, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    combined_audio = torch.cat([mel_tensor, onset_tensor], dim=1).to(CONFIG['DEVICE'])
    length = combined_audio.shape[2]
    pad_len = math.ceil(length / 32) * 32 - length
    if pad_len > 0:
        combined_audio = torch.nn.functional.pad(combined_audio, (0, pad_len))
        
    return combined_audio

# ================= 节奏量化逻辑 =================

def get_best_snap(time_ms, offset, beat_len, divisors):
    rel_time = max(0, time_ms - offset)
    beat_pos = rel_time / beat_len
    
    best_time = time_ms
    min_err = float('inf')
    
    for div in divisors:
        snapped_pos = np.round(beat_pos * div) / div
        candidate_time = offset + snapped_pos * beat_len
        err = abs(candidate_time - time_ms)
        weighted_err = err + (div * 0.05) 
        
        if weighted_err < min_err:
            min_err = weighted_err
            best_time = candidate_time
            
    return best_time, min_err

def quantize_beat_wise(raw_notes, beat_len, offset, target_sr):
    # 节奏模型定义
    divisors_straight = [1, 2, 4]
    if target_sr > 4.5: divisors_straight += [8]
    if target_sr > 6: divisors_straight += [16]
    
    divisors_swing = [1, 3]
    if target_sr > 4: divisors_swing += [6]
    if target_sr > 5.5: divisors_swing += [12]
    
    # 按拍分组
    beat_groups = {}
    note_items = []
    for i, (t, k) in enumerate(raw_notes):
        rel = max(0, t - offset)
        beat_idx = int(rel / beat_len)
        if beat_idx not in beat_groups: beat_groups[beat_idx] = []
        beat_groups[beat_idx].append(t)
        note_items.append({'time': t, 'col': k})
        
    final_snapped_times = {} 
    
    # 小节竞争逻辑
    for beat_idx, times in beat_groups.items():
        if not times: continue
        
        err_straight = sum(get_best_snap(t, offset, beat_len, divisors_straight)[1] for t in times)
        err_swing = sum(get_best_snap(t, offset, beat_len, divisors_swing)[1] for t in times)
        
        bias_factor = 0.8 # 偏向直拍
        active_divisors = divisors_swing if err_swing < err_straight * bias_factor else divisors_straight
            
        for t in times:
            snapped, _ = get_best_snap(t, offset, beat_len, active_divisors)
            final_snapped_times[t] = snapped
            
    processed_objects = []
    for item in note_items:
        t = item['time']
        k = item['col']
        snapped_t = final_snapped_times.get(t, int(t))
        x = CONFIG['COLUMN_X'][k]
        line = f"{x},192,{int(snapped_t)},1,0,0:0:0:0:"
        processed_objects.append((snapped_t, line))
        
    processed_objects.sort(key=lambda x: x[0])
    return [x[1] for x in processed_objects]

def grid_to_hitobjects(grid, beat_len, offset, target_sr, threshold=0.5):
    raw_notes = [] 
    frame_ms = CONFIG['HOP_LENGTH'] / CONFIG['SR'] * 1000
    
    for k in range(4):
        signal = grid[k]
        peaks, _ = scipy.signal.find_peaks(signal, height=threshold, distance=2)
        for p_frame in peaks:
            raw_time = p_frame * frame_ms
            raw_notes.append((raw_time, k))
            
    if not raw_notes: return []
    return quantize_beat_wise(raw_notes, beat_len, offset, target_sr)

def write_osu_file(output_path, audio_filename, hit_objects, sr_val, timing_str):
    content = f"""osu file format v14

[General]
AudioFilename: {audio_filename}
Mode: 3

[Metadata]
Title: AI Generated
Artist: DeepMania
Creator: DeepMania
Version: {sr_val} Stars

[Difficulty]
CircleSize: 4
OverallDifficulty: 8

[TimingPoints]
{timing_str}

[HitObjects]
"""
    content += "\n".join(hit_objects)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved to {output_path}")

# ================= 主入口 =================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepMania: AI Osu!Mania 4K Generator")
    parser.add_argument("--audio", type=str, required=True, help="Path to input MP3 file")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--timing", type=str, required=True, help="Timing point string (e.g. '243,342.85,4...')")
    parser.add_argument("--sr", type=float, default=3.5, help="Target Star Rating (default: 3.5)")
    parser.add_argument("--out", type=str, default="output.osu", help="Output .osu file path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Note detection threshold (0.0-1.0)")
    
    args = parser.parse_args()

    offset, beat_len = parse_manual_timing(args.timing)
    
    model = ManiaUNet(in_channels=4, audio_channels=81, base_dim=64).to(CONFIG['DEVICE'])
    sampler = DiffusionSampler(model, args.model, timesteps=1000)
    
    print(f"Processing {args.audio}...")
    audio_tensor = prepare_audio_with_onset(args.audio)
    
    print(f"Generating for SR={args.sr}...")
    generated_grid = sampler.sample(audio_tensor, args.sr)[0]
    
    print("Applying Rhythm Quantization...")
    hit_objects = grid_to_hitobjects(generated_grid, beat_len, offset, args.sr, threshold=args.threshold)
    
    audio_filename = os.path.basename(args.audio)
    write_osu_file(args.out, audio_filename, hit_objects, args.sr, args.timing)