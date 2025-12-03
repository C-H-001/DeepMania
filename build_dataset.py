import os
import numpy as np
import librosa
import glob
import re
from tqdm import tqdm
from scipy.interpolate import interp1d

# 确保引入你的 sr_calculator
import sr_calculator

# ================= 新配置 =================
CONFIG = {
    'SR': 22050,
    'N_FFT': 2048,  # 增大 FFT 窗口以配合大的 Hop
    'HOP_LENGTH': 512,  # 建议 512 (23ms), 1024 (46ms) 精度太低
    'N_MELS': 80,
    'SLICE_LEN': 256,  # 因为 Hop 变大了，Slice 帧数要减小以保持时长不变
    # 256 * 512 / 22050 ≈ 5.9 秒一个切片
    # 或者设为 512 ≈ 11.8 秒
    'KEYS': 4,
    'OUTPUT_DIR': './processed_dataset_v2'
}


# ===========================================

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
    grid = np.zeros((total_frames, CONFIG['KEYS']), dtype=np.float32)
    for (col, start_time_ms, end_time_ms) in note_seq:
        if col >= CONFIG['KEYS']: continue
        start_frame = int((start_time_ms / 1000.0) / frame_time_sec)
        if start_frame < total_frames:
            grid[start_frame, col] = 1.0
            # 可选：如果想让模型更好地学习长条，可以把长条身体设为 0.5
            # if end_time_ms > 0: ...
    return grid


def process_single_map(osu_path, audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=CONFIG['SR'])
    except:
        return None

    # 计算 Mel
    melspec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=CONFIG['N_FFT'],
        hop_length=CONFIG['HOP_LENGTH'], n_mels=CONFIG['N_MELS']
    )
    melspec = librosa.power_to_db(melspec, ref=np.max).T  # [Time, 80]

    total_frames = melspec.shape[0]
    frame_time_sec = CONFIG['HOP_LENGTH'] / CONFIG['SR']

    try:
        _, K, _, note_seq, _, _, _, _ = sr_calculator.preprocess_file(osu_path, mod="")
        if K != 4: return None
        overall_sr, df_corners = sr_calculator.calculate(osu_path, mod="")
    except:
        return None

    beatmap_grid = create_beatmap_grid(note_seq, total_frames, frame_time_sec)

    # 难度插值
    t_points = df_corners['time'].values / 1000.0
    d_points = df_corners['D'].values
    strain_interpolator = interp1d(t_points, d_points, kind='linear', bounds_error=False, fill_value=0.0)
    local_difficulty = strain_interpolator(np.arange(total_frames) * frame_time_sec)

    # 切片
    slices = []
    num_slices = total_frames // CONFIG['SLICE_LEN']

    for i in range(num_slices):
        start = i * CONFIG['SLICE_LEN']
        end = start + CONFIG['SLICE_LEN']

        # 过滤掉全空的切片 (可选，防止模型学到太多空)
        chart_slice = beatmap_grid[start:end, :]
        if np.sum(chart_slice) < 2:  # 如果这一段几乎没有 Note
            # 只有当 SR 很低时才保留空切片，否则可能是数据错误
            if np.mean(local_difficulty[start:end]) > 1.0:
                continue

        slices.append({
            'mel': melspec[start:end, :].astype(np.float16),
            'chart': chart_slice.astype(np.int8),
            'local_sr': float(np.mean(local_difficulty[start:end]))
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

        full_audio_path = os.path.join(folder, audio_filename)
        if not os.path.exists(full_audio_path):
            # 简单尝试修复路径
            found = False
            for f in os.listdir(folder):
                if f.lower() == audio_filename.lower():
                    full_audio_path = os.path.join(folder, f);
                    found = True;
                    break
            if not found: continue

        slices = process_single_map(osu_path, full_audio_path)
        if slices: batch_data.extend(slices)

        if len(batch_data) >= 1000:
            np.savez_compressed(
                os.path.join(CONFIG['OUTPUT_DIR'], f"data_{batch_idx}.npz"),
                mels=np.array([s['mel'] for s in batch_data]),
                charts=np.array([s['chart'] for s in batch_data]),
                local_srs=np.array([s['local_sr'] for s in batch_data])
            )
            batch_data = []
            batch_idx += 1

    if batch_data:
        np.savez_compressed(
            os.path.join(CONFIG['OUTPUT_DIR'], f"data_{batch_idx}.npz"),
            mels=np.array([s['mel'] for s in batch_data]),
            charts=np.array([s['chart'] for s in batch_data]),
            local_srs=np.array([s['local_sr'] for s in batch_data])
        )


if __name__ == "__main__":
    # 请修改此处路径
    build_dataset(r"E:/osu!/Songs")