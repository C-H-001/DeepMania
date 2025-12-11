import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import numpy as np

from dataset import ManiaDataset
from model_v2 import ManiaDiffuserV2 
from inference_v2 import DiffusionSampler # 假设 Sampler 抽离在 inference 里，或者复制过来

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 24 
LR = 1e-4
EPOCHS = 100
TIMESTEPS = 1000 
DATA_DIR = "./processed_dataset_v3" 
SAVE_DIR = "./checkpoints"

IN_CHANNELS = 8      
AUDIO_CHANNELS = 83  
BASE_DIM = 128       

class DiffusionTrainer:
    # ... (init, cosine_beta_schedule, noise_images, sample_timesteps 保持不变) ...
    def __init__(self, model, timesteps=1000):
        # 复制之前的初始化代码
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

    def train_step(self, x_start, mel, sr, ln_ratio):
        t = self.sample_timesteps(x_start.shape[0])
        x_noisy, noise = self.noise_images(x_start, t)
        
        predicted_noise = self.model(x_noisy, mel, t, sr, ln_ratio)
        
        # 【核心改动】Weighted Loss
        loss = F.mse_loss(predicted_noise, noise, reduction='none')
        
        weights = torch.ones_like(loss)
        # 1. Head (0-3) 权重 x2
        weights[:, :4, :] *= 2.0 
        # 2. Body (4-7) 中是长条的部分 (原图 > 0) 权重 x3
        is_ln = (x_start[:, 4:, :] > 0.0).float()
        weights[:, 4:, :] += is_ln * 2.0 
        
        return (loss * weights).mean()

def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    
    dataset = ManiaDataset(DATA_DIR, sigma=1.5)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    model = ManiaDiffuserV2(in_channels=IN_CHANNELS, audio_channels=AUDIO_CHANNELS, base_dim=BASE_DIM).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    diffusion = DiffusionTrainer(model, TIMESTEPS)
    
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        avg_loss = 0
        
        for audio, chart, sr, ln_ratio in pbar:
            audio = audio.to(DEVICE)
            chart = chart.to(DEVICE)
            sr = sr.to(DEVICE)
            
            # 【核心改动】CFG Dropout (20% 概率设为 -1)
            if np.random.random() < 0.2:
                ln_input = torch.full_like(ln_ratio, -1.0)
            else:
                ln_input = ln_ratio
            ln_input = ln_input.to(DEVICE)
            
            optimizer.zero_grad()
            loss = diffusion.train_step(chart, audio, sr, ln_input)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            avg_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        scheduler.step()
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"v3_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    main()