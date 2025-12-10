import argparse
import torch
import numpy as np
import librosa
import os
import math
import scipy.signal
from tqdm import tqdm

from model_v2 import ManiaDiffuserV2


import sr_calculator

# ================= 配置 =================
CONFIG = {
    'SR': 22050,
    'HOP_LENGTH': 512,
    'N_FFT': 2048,
    'N_MELS': 80,
    'KEYS': 4,
    'DEVICE': "cuda:1" if torch.cuda.is_available() else "cpu",
    'COLUMN_X': [64, 192, 320, 448] 
}

# 【关键配置】物理延迟补偿
TRAIN_SHIFT_FRAMES = 2.5

# ================= 核心类与函数 =================

class DiffusionSampler:
    def __init__(self, model, checkpoint_path, timesteps=1000):
        self.model = model
        self.timesteps = timesteps
        self.device = CONFIG['DEVICE']
        
        print(f"Loading model from {checkpoint_path}...")
        # 加载文件
        loaded_content = torch.load(checkpoint_path, map_location=self.device)
        
        # === 核心修改：自动判断是新版还是旧版格式 ===
        if isinstance(loaded_content, dict) and 'model_state_dict' in loaded_content:
            # 如果是新版 checkpoint (包含 epoch, optimizer 等信息)
            # 我们只提取 'model_state_dict' (模型权重)
            print("Detected Resume-CheckPoint format. Extracting weights...")
            state_dict = loaded_content['model_state_dict']
        else:
            # 如果是旧版 (只包含权重)
            state_dict = loaded_content
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # 使用 Cosine Schedule (与训练保持一致)
        self.beta = self.cosine_beta_schedule(timesteps).to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        # 修正 alpha_hat_prev 的维度问题
        self.alpha_hat_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alpha_hat[:-1]])

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    @torch.no_grad()
    # def sample(self, audio_input, target_sr):
    #     b, c, l = audio_input.shape
    #     # 这里只生成 4 通道 (无长条)
    #     img = torch.randn((b, 4, l), device=self.device)
    #     sr_tensor = torch.tensor([target_sr], dtype=torch.float32).to(self.device)
        
    #     for i in tqdm(reversed(range(0, self.timesteps)), desc="Sampling", total=self.timesteps):
    #         t = torch.full((b,), i, device=self.device, dtype=torch.long)
    #         predicted_noise = self.model(img, audio_input, t, sr_tensor)
            
    #         alpha = self.alpha[t][:, None, None]
    #         alpha_hat = self.alpha_hat[t][:, None, None]
    #         beta = self.beta[t][:, None, None]
            
    #         if i > 0:
    #             noise = torch.randn_like(img)
    #         else:
    #             noise = torch.zeros_like(img)
            
    #         # DDPM 采样公式
    #         img = (1 / torch.sqrt(alpha)) * (img - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            
    #     img = (img.clamp(-1, 1) + 1) / 2
    #     return img.cpu().numpy()

    @torch.no_grad()
    def sample(self, audio_input, target_sr, cfg_scale=4.0): # 推荐 scale 在 3.0 - 7.0 之间
        b, c, l = audio_input.shape
        
        # 初始噪声
        img = torch.randn((b, 4, l), device=self.device)
        
        # 准备两个条件
        # 1. 有条件 (Conditional): 比如 SR=4.0
        sr_cond = torch.tensor([target_sr], dtype=torch.float32).to(self.device)
        # 2. 无条件 (Unconditional): SR=-1.0 (必须与训练时的 drop value 一致)
        sr_uncond = torch.tensor([-1.0], dtype=torch.float32).to(self.device)
        
        print(f"✨ Using Classifier-Free Guidance (Scale={cfg_scale})")
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc="CFG Sampling"):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            
            # === CFG 核心逻辑 ===
            # 我们需要计算两次模型输出，为了节省显存，我们分开计算
            # (如果显存够大，也可以拼在一起算 batch=2)
            
            # 1. 计算 Conditional Noise
            noise_cond = self.model(img, audio_input, t, sr_cond)
            
            # 2. 计算 Unconditional Noise
            noise_uncond = self.model(img, audio_input, t, sr_uncond)
            
            # 3. 混合 (Extrapolate)
            # 这一步会强化 SR 条件带来的差异
            predicted_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            # ===================
            
            # 标准 DDPM 更新步骤 (保持不变)
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
    """解析 timing 字符串"""
    try:
        clean_str = timing_str.replace('"', '').replace("'", "").strip()
        parts = clean_str.split(',')
        offset = float(parts[0])
        beat_len = float(parts[1])
        bpm = 60000.0 / beat_len
        print(f"Manual Timing Parsed: Offset={offset}ms, BPM={bpm:.2f}")
        return offset, beat_len
    except Exception as e:
        print(f"Error parsing timing string: {timing_str}")
        print("Format should be: 'Offset,BeatLength,...'")
        raise e

def calc_onset_feature(mel_db):
    onset = librosa.onset.onset_strength(S=mel_db, sr=CONFIG['SR'])
    if onset.max() > 1e-4:
        onset = onset / onset.max() * 2.0 - 1.0
    else:
        onset = np.zeros_like(onset) - 1.0
    return torch.tensor(onset, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def prepare_audio_with_onset(audio_path):
    """准备音频特征 (Mel + 3 Band Onset) 并进行 Padding"""
    y, sr = librosa.load(audio_path, sr=CONFIG['SR'])
    
    # 1. Mel Spectrogram
    melspec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=CONFIG['N_FFT'], 
        hop_length=CONFIG['HOP_LENGTH'], n_mels=CONFIG['N_MELS']
    )
    log_mel = librosa.power_to_db(melspec, ref=np.max) # [80, Time]
    
    # 2. Multi-band Onset
    # 分割频带
    mel_low = log_mel[0:15, :]
    mel_mid = log_mel[15:50, :]
    mel_high = log_mel[50:, :]
    
    t_low = calc_onset_feature(mel_low)
    t_mid = calc_onset_feature(mel_mid)
    t_high = calc_onset_feature(mel_high)
    
    # Mel Normalization
    norm_mel = log_mel / 40.0 + 1.0
    mel_tensor = torch.tensor(norm_mel, dtype=torch.float32).unsqueeze(0) # [1, 80, T]
    
    # 3. Concat -> [1, 83, T]
    # 注意 t_low, t_mid, t_high 已经是 [1, 1, T] 并在 device 上处理
    combined_audio = torch.cat([mel_tensor, t_low, t_mid, t_high], dim=1).to(CONFIG['DEVICE'])
    
    # 4. Padding (U-Net 要求长度是 32 的倍数)
    length = combined_audio.shape[2]
    pad_len = math.ceil(length / 32) * 32 - length
    if pad_len > 0:
        combined_audio = torch.nn.functional.pad(combined_audio, (0, pad_len))
        
    return combined_audio, length
def get_group_error(times, offset, beat_len, divisors):
    """计算一组音符在特定 rhythm set 下的总拟合误差"""
    total_error = 0
    for t in times:
        rel = t - offset
        beat_pos = rel / beat_len
        min_dist = float('inf')
        for div in divisors:
            snapped_pos = round(beat_pos * div) / div
            candidate_time = offset + snapped_pos * beat_len
            # 距离惩罚 + 复杂度惩罚
            score = abs(candidate_time - t) + (div * 0.1) 
            if score < min_dist:
                min_dist = score
        total_error += min_dist
    return total_error

def snap_time_to_divisors(t, offset, beat_len, divisors):
    """执行吸附"""
    rel = t - offset
    beat_pos = rel / beat_len
    best_time = t
    min_dist = float('inf')
    for div in divisors:
        snapped_pos = round(beat_pos * div) / div
        candidate_time = offset + snapped_pos * beat_len
        dist = abs(candidate_time - t)
        if dist < min_dist:
            min_dist = dist
            best_time = candidate_time
    return best_time

def quantize_measure_wise(raw_notes, beat_len, offset, target_sr):
    """
    以小节(4拍)为单位进行 Straight vs Swing 竞争
    """
    divs_straight = [1, 2, 4]
    divs_swing = [1, 2, 3] # 1/2 是共用的
    
    # 高难度下的扩展
    if target_sr >= 3.5:
        divs_straight += [8]    # Stream
        divs_swing += [6]       # Fast Swing
        
    if target_sr >= 5.8:
        divs_straight += [16]
        divs_swing += [12]

    # 按小节分组
    measure_len = beat_len * 4
    measure_groups = {} 
    note_map = []
    
    for t, k in raw_notes:
        rel = t - offset
        m_idx = int(rel / measure_len)
        if m_idx not in measure_groups: measure_groups[m_idx] = []
        measure_groups[m_idx].append(t)
        note_map.append({'time': t, 'col': k})

    final_snapped_map = {}

    # 小节级竞争
    SWING_THRESHOLD_RATIO = 0.65
    
    for m_idx, times in measure_groups.items():
        if not times: continue
        
        # 只取 Beat 中间的复杂音符进行投票
        complex_notes = []
        for t in times:
            rel = (t - offset) / beat_len
            if abs(rel - round(rel)) > 0.1:
                complex_notes.append(t)
        
        if not complex_notes:
            active_divs = divs_straight
        else:
            err_str = get_group_error(complex_notes, offset, beat_len, divs_straight)
            err_swg = get_group_error(complex_notes, offset, beat_len, divs_swing)
            
            if err_swg < err_str * SWING_THRESHOLD_RATIO:
                active_divs = divs_swing
            else:
                active_divs = divs_straight

        for t in times:
            snapped = snap_time_to_divisors(t, offset, beat_len, active_divs)
            final_snapped_map[t] = snapped

    processed_objects = []
    seen = set()
    
    for item in note_map:
        raw_t = item['time']
        k = item['col']
        snapped_t = final_snapped_map.get(raw_t, int(raw_t))
        
        # 再次取整
        snapped_t = int(round(snapped_t))
        if snapped_t < 0: continue
        
        x = CONFIG['COLUMN_X'][k]
        key = (snapped_t, x)
        
        if key not in seen:
            seen.add(key)
            # 输出 Rice Note
            line = f"{x},192,{snapped_t},1,0,0:0:0:0:"
            processed_objects.append((snapped_t, line))
            
    processed_objects.sort(key=lambda x: x[0])
    return [x[1] for x in processed_objects]

def grid_to_hitobjects(grid, beat_len, offset, target_sr, threshold=0.5):
    raw_notes = [] 
    frame_ms = CONFIG['HOP_LENGTH'] / CONFIG['SR'] * 1000
    
    # 【关键修正】物理延迟补偿
    # Offset Correction = 2.8 * 23.2ms ≈ 65ms
    offset_correction = TRAIN_SHIFT_FRAMES * frame_ms
    
    for k in range(4):
        signal = grid[k]
        # 寻找波峰
        peaks, _ = scipy.signal.find_peaks(signal, height=threshold, distance=2)
        
        for p_frame in peaks:
            # 原始时间 = 帧索引时间 - 延迟补偿
            raw_time = p_frame * frame_ms - offset_correction
            raw_notes.append((raw_time, k))
            
    if not raw_notes: return []
    
    # 进入节奏量化流程
    return quantize_measure_wise(raw_notes, beat_len, offset, target_sr)

def write_osu_file(output_path, audio_filename, hit_objects, sr_val, timing_str):
    timing_str = timing_str.replace('"', '').strip()
    content = f"""osu file format v14

[General]
AudioFilename: {audio_filename}
AudioLeadIn: 0
PreviewTime: -1
Countdown: 0
SampleSet: Soft
StackLeniency: 0.7
Mode: 3
LetterboxInBreaks: 0
WidescreenStoryboard: 0

[Metadata]
Title: AI Generated
TitleUnicode: AI Generated
Artist: DeepMania
ArtistUnicode: DeepMania
Creator: AI
Version: {sr_val} Stars
Source:
Tags:
BeatmapID: 0
BeatmapSetID: 0

[Difficulty]
HPDrainRate: 8
CircleSize: 4
OverallDifficulty: 8
ApproachRate: 5
SliderMultiplier: 1.4
SliderTickRate: 1

[Events]
//Background and Video events
//Break Periods
//Storyboard Layer 0 (Background)
//Storyboard Layer 1 (Fail)
//Storyboard Layer 2 (Pass)
//Storyboard Layer 3 (Foreground)
//Storyboard Sound Samples

[TimingPoints]
{timing_str}

[HitObjects]
""" 
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
        for line in hit_objects:
            f.write(line + "\n")
            
    print(f"Saved beatmap to {output_path}")

# ================= 主入口 =================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepMania Inference")
    parser.add_argument("--audio", type=str, required=True, help="Path to input MP3")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--timing", type=str, required=True, help="Timing string 'Offset,BeatLen,4...'")
    parser.add_argument("--sr", type=float, default=3.5, help="Target Star Rating")
    parser.add_argument("--out", type=str, default="output.osu", help="Output filename")
    parser.add_argument("--threshold", type=float, default=0.55, help="Note detection threshold")
    parser.add_argument("--base_dim", type=int, default=128, help="Model Base Dim (64 or 128)")
    parser.add_argument("--cfg", type=float, default=4.0, help="CFG Scale (Higher = stricter adherence to SR)")
    args = parser.parse_args()

    offset, beat_len = parse_manual_timing(args.timing)
    
    # === 修改此处：audio_channels 设为 83 ===
    model = ManiaDiffuserV2(in_channels=4, audio_channels=83, base_dim=args.base_dim).to(CONFIG['DEVICE'])
    
    sampler = DiffusionSampler(model, args.model, timesteps=1000)
    
    print(f"Processing {args.audio}...")
    # 这里调用的是新的 prepare_audio_with_onset
    audio_tensor, original_len = prepare_audio_with_onset(args.audio)

    print(f"Generating for SR={args.sr} with CFG={args.cfg}...")
    generated_grid = sampler.sample(audio_tensor, args.sr, cfg_scale=args.cfg)[0]
    
    generated_grid = generated_grid[:, :original_len]
    
    print("Applying Quantization...")
    hit_objects = grid_to_hitobjects(generated_grid, beat_len, offset, args.sr, threshold=args.threshold)
    
    temp_output_path = args.out 
    audio_filename = os.path.basename(args.audio)
    write_osu_file(temp_output_path, audio_filename, hit_objects, args.sr, args.timing)
    
    print("-" * 30)
    print("Calculating Actual Star Rating...")
    try:
        real_sr, _ = sr_calculator.calculate(temp_output_path, mod="")
        print(f"🎯 Target SR: {args.sr}")
        print(f"📊 Actual SR: {real_sr:.2f}")

        base, ext = os.path.splitext(temp_output_path)
        new_filename = f"{base}_Real{real_sr:.2f}{ext}"
        if os.path.exists(new_filename):
            os.remove(new_filename)
        os.rename(temp_output_path, new_filename)
        print(f"✅ Renamed to: {new_filename}")
    except Exception as e:
        print(f"SR Calc failed (maybe not enough notes): {e}")