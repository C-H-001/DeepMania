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
import argparse

# 引入项目模块
from dataset import ManiaDataset
from model_v2 import ManiaDiffuserV2 

# ================= 配置区域 =================
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 100
TIMESTEPS = 1000 
DATA_DIR = "./processed_dataset" 
SAVE_DIR = "./checkpoints"

# 通道配置 (83 = 80 Mel + 3 Onsets)
IN_CHANNELS = 4      
AUDIO_CHANNELS = 83  
BASE_DIM = 128       

# ===========================================

class DiffusionTrainer:
    def __init__(self, model, timesteps=1000):
        self.model = model
        self.timesteps = timesteps
        self.device = DEVICE
        self.beta = self.cosine_beta_schedule(timesteps).to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat)

    def cosine_beta_schedule(self, timesteps, s=0.008):
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

    # def train_step(self, x_start, mel, sr):
    #     t = self.sample_timesteps(x_start.shape[0])
    #     x_noisy, noise = self.noise_images(x_start, t)
    #     predicted_noise = self.model(x_noisy, mel, t, sr)
    #     loss = F.mse_loss(predicted_noise, noise)
    #     return loss

    def train_step(self, x_start, mel, sr, prob_uncond=0.15): # 15% 的概率丢弃 SR 条件
        t = self.sample_timesteps(x_start.shape[0])
        x_noisy, noise = self.noise_images(x_start, t)
        
        # 克隆一下 SR，避免修改原始数据
        sr_input = sr.clone()
        
        # 生成掩码：15% 的概率为 True
        if prob_uncond > 0:
            mask = torch.rand(sr.shape[0], device=self.device) < prob_uncond
            # 将被选中的 SR 设为 -1.0 (代表 Unconditional)
            # 注意：Linear层可以接受负数，模型会自动学习 -1 代表“空”
            sr_input[mask] = -1.0
        
        predicted_noise = self.model(x_noisy, mel, t, sr_input)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

# === 验证监控器 (简化版) ===
class ValidationMonitor:
    def __init__(self, diffusion_trainer, device):
        self.diffusion = diffusion_trainer
        self.device = device

    @torch.no_grad()
    def check(self, model, mel, sr, epoch):
        model.eval()
        # 简单打印日志，避免刷屏
        print(f"\n[Epoch {epoch}] Saving checkpoint and checking distribution...")
        model.train()

def main(args):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    print(f"Using device: {DEVICE}")
    
    # 1. 数据准备
    dataset = ManiaDataset(DATA_DIR, sigma=1.5)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # 2. 模型初始化
    model = ManiaDiffuserV2(
        in_channels=IN_CHANNELS,       
        audio_channels=AUDIO_CHANNELS, 
        base_dim=BASE_DIM,             
        dim_mults=(1, 2, 4, 8)         
    ).to(DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    diffusion = DiffusionTrainer(model, timesteps=TIMESTEPS)
    monitor = ValidationMonitor(diffusion, DEVICE)

    start_epoch = 0

    # ================= 续训逻辑 (Resume Logic) =================
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"🔄 Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=DEVICE)
            
            # 判断是旧版(只存了权重)还是新版(存了完整状态)
            if 'model_state_dict' in checkpoint:
                # 完整状态加载
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"✅ Resumed from Epoch {start_epoch}")
            else:
                # 旧版兼容：只加载权重，优化器重置
                model.load_state_dict(checkpoint)
                print(f"⚠️ Loaded weights only (Old Format). Restarting scheduler from Epoch 0.")
        else:
            print(f"❌ Checkpoint not found: {args.resume}")
            return
    # ==========================================================

    # 3. 训练循环
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        avg_loss = 0
        
        for audio, chart, sr in pbar:
            audio = audio.to(DEVICE)
            chart = chart.to(DEVICE)
            sr = sr.to(DEVICE)
            
            optimizer.zero_grad()
            loss = diffusion.train_step(chart, audio, sr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            avg_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step()
        
        # 保存逻辑：保存完整状态字典
        if (epoch + 1) % 5 == 0 or epoch == 0: # 每5轮保存一次
            save_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pt")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss / len(dataloader)
            }, save_path)
            
            # 另外存一个只含权重的 best.pt 方便推理脚本直接读取
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best.pt"))
            
            monitor.check(model, None, None, epoch+1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 添加 --resume 参数，如果不传则从头开始
    parser.add_argument("--resume", type=str, default='checkpoints/checkpoint_epoch_25.pt', help="Path to checkpoint (.pt) to resume from")
    args = parser.parse_args()
    
    main(args)