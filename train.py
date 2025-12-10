import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import numpy as np
import math

# 引入之前的类
from dataset import ManiaDataset
# 注意：这里引入的是 v2 版本的 U-Net
from model_v2 import ManiaDiffuserV2 

# ================= 配置区域 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 显存警告：V2 模型较大。
# 如果 12G 显存，建议 Batch Size 8-12
# 如果 24G 显存，建议 Batch Size 24-32
BATCH_SIZE = 32 

LR = 1e-4
EPOCHS = 100
TIMESTEPS = 1000 # Diffusion 步数
DATA_DIR = "./processed_dataset" 
SAVE_DIR = "./checkpoints"

# 通道配置 (必须与 dataset.py 和 model_v2.py 一致)
IN_CHANNELS = 4      # 4 Head + 4 Tail
AUDIO_CHANNELS = 83  # 80 Mel + 1 Onset
BASE_DIM = 128       # 模型宽度
# ===========================================

# === 验证监控器 ===
class ValidationMonitor:
    def __init__(self, diffusion_trainer, device):
        self.diffusion = diffusion_trainer
        self.device = device

    @torch.no_grad()
    def check(self, model, mel, sr, epoch):
        """生成预览以检查是否发生模式崩塌"""
        model.eval()
        print(f"\n[Epoch {epoch}] Generating preview...")

        # 生成形状: [1, 8, Length]
        l = mel.shape[2]
        img = torch.randn((1, IN_CHANNELS, l), device=self.device)
        
        # 完整采样
        for i in reversed(range(0, self.diffusion.timesteps)):
            t = torch.full((1,), i, device=self.device, dtype=torch.long)
            predicted_noise = model(img, mel, t, sr)
            
            alpha = self.diffusion.alpha[t][:, None, None]
            alpha_hat = self.diffusion.alpha_hat[t][:, None, None]
            beta = self.diffusion.beta[t][:, None, None]
            
            if i > 0:
                noise = torch.randn_like(img)
            else:
                noise = torch.zeros_like(img)
                
            img = (1 / torch.sqrt(alpha)) * (img - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        # 统计分布
        img = (img.clamp(-1, 1) + 1) / 2
        grid = img[0].cpu().numpy() # [8, Length]
        
        # 这里的 grid 是热力图，我们简单统计一下强度
        # 只统计前4个通道 (Heads)
        heads = grid[:4, :]
        threshold = 0.5
        notes = (heads > threshold).astype(int)
        counts = np.sum(notes, axis=1)
        total_notes = np.sum(counts) + 1e-8
        
        print("-" * 40)
        print(f"📊 Distribution (Target SR: {sr.item():.1f})")
        
        max_notes = np.max(counts)
        for k in range(4):
            count = counts[k]
            ratio = count / total_notes * 100
            bar_len = int((count / max_notes) * 20) if max_notes > 0 else 0
            bar = "█" * bar_len
            print(f"   Col {k}: {count:5d} ({ratio:5.1f}%) | {bar}")
            
        # 简单检查长条尾巴分布
        tails_count = np.sum((grid[4:, :] > threshold).astype(int))
        print(f"   LN Tails Detected: {tails_count}")
            
        print("-" * 40)
        model.train()

# === 扩散模型辅助类 ===
class DiffusionTrainer:
    def __init__(self, model, timesteps=1000):
        self.model = model
        self.timesteps = timesteps
        self.device = DEVICE
        
        # 使用 Cosine Schedule (对图像/Grid生成效果更好)
        self.beta = self.cosine_beta_schedule(timesteps).to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def noise_images(self, x, t):
        sqrt_alpha_hat = self.sqrt_alpha_hat[t][:, None, None]
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[t][:, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.timesteps, size=(n,), device=self.device)

    def train_step(self, x_start, mel, sr):
        t = self.sample_timesteps(x_start.shape[0])
        x_noisy, noise = self.noise_images(x_start, t)
        
        # V2 模型的前向传播
        predicted_noise = self.model(x_noisy, mel, t, sr)
        
        # 使用 MSE Loss
        loss = F.mse_loss(predicted_noise, noise)
        return loss

# === 主训练循环 ===
def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    print(f"Using device: {DEVICE}")
    print(f"Model V2 Config: BaseDim={BASE_DIM}, InChannels={IN_CHANNELS}")
    
    # 1. 数据准备
    # sigma=1.0 或 1.5 比较合适，太大会导致相邻音符粘连
    dataset = ManiaDataset(DATA_DIR, sigma=1.5)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # 2. 模型初始化 (V2 U-Net)
    model = ManiaDiffuserV2(
        in_channels=IN_CHANNELS,       # 8
        audio_channels=AUDIO_CHANNELS, # 81
        base_dim=BASE_DIM,             # 128
        dim_mults=(1, 2, 4, 8)         # 深度配置
    ).to(DEVICE)
    
    # 优化器与调度器
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS) # 学习率余弦衰减
    
    diffusion = DiffusionTrainer(model, timesteps=TIMESTEPS)
    monitor = ValidationMonitor(diffusion, DEVICE)
    
    # 抽取验证样本
    sample_data = dataset[0] 
    fixed_audio = sample_data[0].unsqueeze(0).to(DEVICE) # [1, 81, L]
    fixed_sr = sample_data[2].unsqueeze(0).to(DEVICE)
    print(f"Validation loaded. Target SR: {fixed_sr.item():.2f}")

    # 3. 训练
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        avg_loss = 0
        
        for audio, chart, sr in pbar:
            audio = audio.to(DEVICE)
            chart = chart.to(DEVICE)
            sr = sr.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 计算 Loss
            loss = diffusion.train_step(chart, audio, sr)
            
            loss.backward()
            
            # 【重要】梯度裁剪：防止大模型训练初期梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            avg_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
        
        # 更新学习率
        scheduler.step()
        
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss / len(dataloader):.6f}")
        
        # 监控与保存
        if (epoch + 1) % 5 == 0 or epoch == 0:
            monitor.check(model, fixed_audio, fixed_sr, epoch+1)
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"model_v2_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    main()