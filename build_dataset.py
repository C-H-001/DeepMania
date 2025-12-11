import os
import numpy as np
import librosa
import glob
import re
from tqdm import tqdm
from scipy.interpolate import interp1d
import sr_calculator

CONFIG = {
    'SR': 22050,
    'N_FFT': 2048,
    'HOP_LENGTH': 512,
    'N_MELS': 80,
    'SLICE_LEN': 256,
    'KEYS': 4,
    'OUTPUT_DIR': './processed_dataset_v3' # 建议用新目录
}

def get_audio_filename_from_osu(osu_path):
    try:
        with open(osu_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            match = re.search(r'AudioFilename\s*:\s*(.*)', content)
            if match: return match.group(1).strip()
    except:
        pass
    return None

def create_beatmap_grid(note_seq, total_frames, frame_time_sec):
    # 【改动】输出 8 通道: 0-3 Head, 4-7 Body
    grid = np.zeros((total_frames, CONFIG['KEYS'] * 2), dtype=np.float32)
    
    for (col, start_time_ms, end_time_ms) in note_seq:
        if col >= CONFIG['KEYS']: continue
        
        # 1. Head
        start_frame = int((start_time_ms / 1000.0) / frame_time_sec)
        if start_frame < total_frames:
            grid[start_frame, col] = 1.0
            
            # 2. Body (如果是长条)
            if end_time_ms > 0:
                end_frame = int((end_time_ms / 1000.0) / frame_time_sec)
                end_frame = min(end_frame, total_frames)
                if end_frame > start_frame:
                    # 填充对应的 Body 通道 (Offset +4)
                    grid[start_frame:end_frame, col + CONFIG['KEYS']] = 1.0
    return grid

def process_single_map(osu_path, audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=CONFIG['SR'])
    except: return None

    # Mel 谱 (只存基础特征，Onset 在 dataset.py 里动态算)
    melspec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=CONFIG['N_FFT'],
        hop_length=CONFIG['HOP_LENGTH'], n_mels=CONFIG['N_MELS']
    )
    melspec = librosa.power_to_db(melspec, ref=np.max).T 

    total_frames = melspec.shape[0]
    frame_time_sec = CONFIG['HOP_LENGTH'] / CONFIG['SR']

    try:
        _, K, _, note_seq, _, _, _, _ = sr_calculator.preprocess_file(osu_path, mod="")
        if K != 4: return None
        overall_sr, df_corners = sr_calculator.calculate(osu_path, mod="")
    except: return None

    beatmap_grid = create_beatmap_grid(note_seq, total_frames, frame_time_sec)

    # 难度插值
    t_points = df_corners['time'].values / 1000.0
    d_points = df_corners['D'].values
    strain_interpolator = interp1d(t_points, d_points, kind='linear', bounds_error=False, fill_value=0.0)
    local_difficulty = strain_interpolator(np.arange(total_frames) * frame_time_sec)

    slices = []
    num_slices = total_frames // CONFIG['SLICE_LEN']

    for i in range(num_slices):
        start = i * CONFIG['SLICE_LEN']
        end = start + CONFIG['SLICE_LEN']
        
        chart_slice = beatmap_grid[start:end, :]
        
        # 【改动】计算 LN Ratio
        # 统计有多少个 Note 是长条 (通过检查 Body 通道是否有能量)
        heads = chart_slice[:, :4]
        bodies = chart_slice[:, 4:]
        
        total_notes = 0
        ln_notes = 0
        
        for k in range(4):
            # 找到 Head 的位置
            head_indices = np.where(heads[:, k] > 0.5)[0]
            for h_idx in head_indices:
                total_notes += 1
                # 检查该 Head 后续是否有 Body
                # 简单判定：Head 所在帧或下一帧，Body 通道是否为 1
                is_ln = False
                if h_idx < len(chart_slice):
                     if bodies[h_idx, k] > 0.5: is_ln = True
                if h_idx + 1 < len(chart_slice):
                     if bodies[h_idx+1, k] > 0.5: is_ln = True
                
                if is_ln: ln_notes += 1
        
        ln_ratio = ln_notes / max(1, total_notes) if total_notes > 0 else 0.0

        # 过滤空且低难度的片段
        if np.sum(heads) < 2 and np.mean(local_difficulty[start:end]) < 1.0:
            continue

        slices.append({
            'mel': melspec[start:end, :].astype(np.float16),
            'chart': chart_slice.astype(np.int8),
            'local_sr': float(np.mean(local_difficulty[start:end])),
            'ln_ratio': float(ln_ratio) # 保存比例
        })

    return slices

def build_dataset(root_folder):
    if not os.path.exists(CONFIG['OUTPUT_DIR']): os.makedirs(CONFIG['OUTPUT_DIR'])
    osu_files = glob.glob(os.path.join(root_folder, "**/*.osu"), recursive=True)
    batch_data = []
    batch_idx = 0

    print("Starting processing...")
    for osu_path in tqdm(osu_files):
        folder = os.path.dirname(osu_path)
        audio_filename = get_audio_filename_from_osu(osu_path)
        if not audio_filename: continue
        
        # 简单的路径修复
        full_audio_path = os.path.join(folder, audio_filename)
        if not os.path.exists(full_audio_path):
             for f in os.listdir(folder):
                if f.lower() == audio_filename.lower():
                    full_audio_path = os.path.join(folder, f); break
        if not os.path.exists(full_audio_path): continue

        slices = process_single_map(osu_path, full_audio_path)
        if slices: batch_data.extend(slices)

        if len(batch_data) >= 2000: # 稍微加大 Batch 减少文件碎片
            np.savez_compressed(
                os.path.join(CONFIG['OUTPUT_DIR'], f"data_{batch_idx}.npz"),
                mels=np.array([s['mel'] for s in batch_data]),
                charts=np.array([s['chart'] for s in batch_data]),
                local_srs=np.array([s['local_sr'] for s in batch_data]),
                ln_ratios=np.array([s['ln_ratio'] for s in batch_data]) # 保存
            )
            batch_data = []
            batch_idx += 1
            
    # Save remaining
    if batch_data:
        np.savez_compressed(
            os.path.join(CONFIG['OUTPUT_DIR'], f"data_{batch_idx}.npz"),
            mels=np.array([s['mel'] for s in batch_data]),
            charts=np.array([s['chart'] for s in batch_data]),
            local_srs=np.array([s['local_sr'] for s in batch_data]),
            ln_ratios=np.array([s['ln_ratio'] for s in batch_data])
        )

if __name__ == "__main__":
    build_dataset(r"E:/osu!/Songs") # 修改你的路径