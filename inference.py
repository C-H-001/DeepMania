import argparse
import torch
import numpy as np
import librosa
import os
import math
import scipy.signal
from tqdm import tqdm
from model import ManiaUNet
import sr_calculator
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

# 【关键配置】必须与 dataset.py 中的 SHIFT_FRAMES 保持一致
# 3帧 * 23.2ms ≈ 70ms 的物理延迟补偿
TRAIN_SHIFT_FRAMES = 2.8 

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
        # 修正 alpha_hat_prev 的计算，防止维度错误
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
            
            # 标准 DDPM 采样公式
            img = (1 / torch.sqrt(alpha)) * (img - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            
        img = (img.clamp(-1, 1) + 1) / 2
        return img.cpu().numpy()

def parse_manual_timing(timing_str):
    """解析 timing 字符串，处理潜在的引号或空格问题"""
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
        print("Format should be: 'Offset,BeatLength,...' (e.g. '1020,500,4...')")
        raise e

def prepare_audio_with_onset(audio_path):
    y, sr = librosa.load(audio_path, sr=CONFIG['SR'])
    
    # 1. Mel Spectrogram
    melspec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=CONFIG['N_FFT'], 
        hop_length=CONFIG['HOP_LENGTH'], n_mels=CONFIG['N_MELS']
    )
    log_mel = librosa.power_to_db(melspec, ref=np.max)
    
    # 2. Onset Strength
    onset_env = librosa.onset.onset_strength(S=log_mel, sr=sr)
    
    # Onset Normalization [-1, 1]
    if onset_env.max() > 0:
        onset_env = onset_env / onset_env.max() * 2.0 - 1.0
    else:
        onset_env = np.zeros_like(onset_env) - 1.0
    
    # Mel Normalization [-1, 1]
    norm_mel = log_mel / 40.0 + 1.0
    
    # 3. Concat & Pad
    mel_tensor = torch.tensor(norm_mel, dtype=torch.float32).unsqueeze(0)
    onset_tensor = torch.tensor(onset_env, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    combined_audio = torch.cat([mel_tensor, onset_tensor], dim=1).to(CONFIG['DEVICE'])
    length = combined_audio.shape[2]
    pad_len = math.ceil(length / 32) * 32 - length
    if pad_len > 0:
        combined_audio = torch.nn.functional.pad(combined_audio, (0, pad_len))
        
    return combined_audio

# ================= 节奏量化逻辑 (核心修改) =================

def get_best_snap(time_ms, offset, beat_len, divisors):
    # 计算相对于最近红线的偏移
    rel_time = time_ms - offset
    
    # 如果在红线之前，不做量化直接返回 (或量化到负数拍)
    # 这里简单处理：允许负数，保持数学连续性
    beat_pos = rel_time / beat_len
    
    best_time = time_ms
    min_err = float('inf')
    
    for div in divisors:
        snapped_pos = np.round(beat_pos * div) / div
        candidate_time = offset + snapped_pos * beat_len
        err = abs(candidate_time - time_ms)
        
        # 惩罚项：优先吸附到简单的节拍 (分母越小惩罚越小)
        # div=1 -> penalty=0.05
        # div=4 -> penalty=0.2
        # div=16 -> penalty=0.8
        weighted_err = err + (div * 0.05) 
        
        if weighted_err < min_err:
            min_err = weighted_err
            best_time = candidate_time
            
    return best_time, min_err

# ================= 升级版：小节锁定量化 =================

def get_group_error(times, offset, beat_len, divisors):
    """计算一组音符在特定 rhythm set 下的总拟合误差"""
    total_error = 0
    for t in times:
        # 计算相对位置
        rel = t - offset
        beat_pos = rel / beat_len
        
        min_dist = float('inf')
        for div in divisors:
            # 找最近的网格
            snapped_pos = round(beat_pos * div) / div
            candidate_time = offset + snapped_pos * beat_len
            dist = abs(candidate_time - t)
            
            # 距离惩罚 + 复杂度惩罚 (div越小越好)
            # 我们希望尽量吸附到 1/1, 1/2, 1/4
            score = dist + (div * 0.1) 
            if score < min_dist:
                min_dist = score
        total_error += min_dist
    return total_error

def snap_time_to_divisors(t, offset, beat_len, divisors):
    """执行具体的吸附操作"""
    rel = t - offset
    beat_pos = rel / beat_len
    
    best_time = t
    min_dist = float('inf')
    
    for div in divisors:
        snapped_pos = round(beat_pos * div) / div
        candidate_time = offset + snapped_pos * beat_len
        
        # 这里只看纯物理距离，因为上面已经决定了用哪套divisors
        dist = abs(candidate_time - t)
        if dist < min_dist:
            min_dist = dist
            best_time = candidate_time
            
    return best_time

def quantize_measure_wise(raw_notes, beat_len, offset, target_sr):
    """
    以小节(4拍)为单位进行一致性锁定。
    解决 1/3 和 1/4 混用的问题。
    """
    # --- 1. 定义节奏集合 ---
    # 基础 Straight
    divs_straight = [1, 2, 4]
    # 基础 Swing
    divs_swing = [1, 2, 3] # 1/2 是共用的，但这不影响区分
    
    # 高难度下的扩展
    if target_sr >= 5.5:
        divs_straight += [8]    # Stream
        divs_swing += [6]       # Fast Swing
        
    if target_sr >= 6:
        divs_straight += [16]
        divs_swing += [12]

    print(f"Quantizer: Straight={divs_straight}, Swing={divs_swing}")

    # --- 2. 按小节分组 (假设 4/4 拍) ---
    measure_len = beat_len * 4
    measure_groups = {} # key: measure_index, value: list of raw_times
    note_map = []       # store (time, col)
    
    for t, k in raw_notes:
        rel = t - offset
        # 向下取整到第几个小节
        m_idx = int(rel / measure_len)
        if m_idx not in measure_groups: measure_groups[m_idx] = []
        measure_groups[m_idx].append(t)
        note_map.append({'time': t, 'col': k})

    final_snapped_map = {}

    # --- 3. 小节级竞争 (Winner Takes All) ---
    # 这里是核心：Swing Bias
    # 只有当 SwingError < StraightError * 0.65 时，才切 Swing
    # 意味着 Swing 必须比 Straight 准很多才行
    SWING_THRESHOLD_RATIO = 0.65
    
    for m_idx, times in measure_groups.items():
        if not times: continue
        
        # 排除掉那些明显是 1/1 或 1/2 的音符（它们对分辨节奏没帮助，是干扰项）
        # 我们只关注那些落在 beat 中间的音符
        complex_notes = []
        for t in times:
            rel = (t - offset) / beat_len
            # 如果离整数拍很近 (<0.1拍)，说明是正拍，不参与投票
            if abs(rel - round(rel)) > 0.1:
                complex_notes.append(t)
        
        # 如果一个小节里全是正拍(1/1)，那默认为 Straight
        if not complex_notes:
            active_divs = divs_straight
        else:
            # 计算这一组复杂音符的误差
            err_str = get_group_error(complex_notes, offset, beat_len, divs_straight)
            err_swg = get_group_error(complex_notes, offset, beat_len, divs_swing)
            
            # 判定
            if err_swg < err_str * SWING_THRESHOLD_RATIO:
                active_divs = divs_swing
                # debug_info = "SWING (Locked)"
            else:
                active_divs = divs_straight
                # debug_info = "STRAIGHT"
            
            # print(f"Measure {m_idx}: StrErr={err_str:.1f}, SwgErr={err_swg:.1f} -> {debug_info}")

        # --- 4. 执行吸附 ---
        # 对该小节内的所有音符，强制使用选定的 active_divs
        for t in times:
            snapped = snap_time_to_divisors(t, offset, beat_len, active_divs)
            final_snapped_map[t] = snapped

    # --- 5. 组装输出 ---
    processed_objects = []
    seen = set()
    
    for item in note_map:
        raw_t = item['time']
        k = item['col']
        snapped_t = final_snapped_map.get(raw_t, int(raw_t))
        
        # 再次取整保证整数
        snapped_t = int(round(snapped_t))
        if snapped_t < 0: continue
        
        x = CONFIG['COLUMN_X'][k]
        key = (snapped_t, x)
        
        if key not in seen:
            seen.add(key)
            line = f"{x},192,{snapped_t},1,0,0:0:0:0:"
            processed_objects.append((snapped_t, line))
            
    processed_objects.sort(key=lambda x: x[0])
    return [x[1] for x in processed_objects]
def grid_to_hitobjects(grid, beat_len, offset, target_sr, threshold=0.5):
    raw_notes = [] 
    frame_ms = CONFIG['HOP_LENGTH'] / CONFIG['SR'] * 1000
    
    # 【关键修正】计算需要扣除的物理时间偏移量
    # 修正 Librosa Padding 导致的整体向右偏移
    offset_correction = TRAIN_SHIFT_FRAMES * frame_ms
    
    for k in range(4):
        signal = grid[k]
        # height: 峰值高度
        # distance: 最小间隔帧数 (2帧约为46ms，防止重叠)
        peaks, _ = scipy.signal.find_peaks(signal, height=threshold, distance=2)
        
        for p_frame in peaks:
            # 原始时间 = 帧索引时间 - 训练时的人为偏移
            raw_time = p_frame * frame_ms - offset_correction
            raw_notes.append((raw_time, k))
            
    if not raw_notes: return []
    
    # 进入节奏量化流程
    return quantize_measure_wise(raw_notes, beat_len, offset, target_sr)

def write_osu_file(output_path, audio_filename, hit_objects, sr_val, timing_str):
    # 清理 timing string 的格式
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
    parser = argparse.ArgumentParser(description="DeepMania: AI Osu!Mania 4K Generator")
    parser.add_argument("--audio", type=str, required=True, help="Path to input MP3 file")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--timing", type=str, required=True, help="Timing point string (e.g. '100,333.33,4...')")
    parser.add_argument("--sr", type=float, default=3.5, help="Target Star Rating (default: 3.5)")
    parser.add_argument("--out", type=str, default="output.osu", help="Output .osu file path")
    parser.add_argument("--threshold", type=float, default=0.55, help="Note detection threshold (0.0-1.0)")
    
    args = parser.parse_args()

    # 1. 解析 Timing
    offset, beat_len = parse_manual_timing(args.timing)
    
    # 2. 初始化模型 (确保是 81 通道)
    model = ManiaUNet(in_channels=4, audio_channels=81, base_dim=64).to(CONFIG['DEVICE'])
    sampler = DiffusionSampler(model, args.model, timesteps=1000)
    
    # 3. 准备音频
    print(f"Processing {args.audio}...")
    audio_tensor = prepare_audio_with_onset(args.audio)
    
    # 4. 生成 Grid
    print(f"Generating for SR={args.sr}...")
    generated_grid = sampler.sample(audio_tensor, args.sr)[0]
    
    # 5. 后处理 (含 Shift Correction 和 Rhythm Competition)
    print("Applying Rhythm Quantization...")
    hit_objects = grid_to_hitobjects(generated_grid, beat_len, offset, args.sr, threshold=args.threshold)
    
    # 6. 写入文件 (先保存一个临时文件名)
    audio_filename = os.path.basename(args.audio)
    # 使用 args.out 作为初始路径
    temp_output_path = args.out 
    write_osu_file(temp_output_path, audio_filename, hit_objects, args.sr, args.timing)
    
    # 【新增】计算实际 SR 并重命名
    # ==========================================
    print("-" * 30)
    print("Calculating Actual Star Rating...")
    

    # 调用 sr_calculator
    # 注意：根据之前的修改，你的 calculate 函数应该返回 (sr, df_corners)
    # 我们只需要第一个返回值
    real_sr, _ = sr_calculator.calculate(temp_output_path, mod="")
    
    print(f"🎯 Target SR: {args.sr}")
    print(f"📊 Actual SR: {real_sr:.2f}")

    # 可选：根据实际 SR 重命名文件
    # 例如: output.osu -> output_3.52.osu
    base, ext = os.path.splitext(temp_output_path)
    # 为了防止文件名重复，保留一部分原名
    new_filename = f"{base}_Real{real_sr:.2f}{ext}"
    
    # 重命名文件
    if os.path.exists(new_filename):
        os.remove(new_filename) # 覆盖旧的
    os.rename(temp_output_path, new_filename)
    print(f"✅ Renamed file to: {new_filename}")
    