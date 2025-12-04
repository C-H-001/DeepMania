import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import numpy as np

# 引入之前的类
from dataset import ManiaDataset
from model import ManiaUNet

# ================= 配置区域 =================
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32 
LR = 1e-4
EPOCHS = 100
TIMESTEPS = 1200
DATA_DIR = "./processed_dataset" # 确保这里指向你新的数据集路径
SAVE_DIR = "./checkpoints"
# ===========================================

# === 新增：验证监控器 ===
class ValidationMonitor:
    def __init__(self, diffusion_trainer, device):
        self.diffusion = diffusion_trainer
        self.device = device

    @torch.no_grad()
    def check(self, model, mel, sr, epoch):
        """
        执行一次采样并统计分布
        mel: [1, 80, L]
        sr: [1, 1]
        """
        model.eval()
        print(f"\n[Epoch {epoch}] 正在生成预览以检查轨道分布...")

        # 1. 执行采样 (简化版，为了速度可以使用较少的步数，或者坚持1000步以求准确)
        # 这里直接调用 DiffusionTrainer 里的采样逻辑（我们需要先把它加进去）
        # 如果 Trainer 里没有 sample 方法，我们在这里手动写一个简化的
        
        b, c, l = 4, 4, mel.shape[2] # 这里的 b=4 是指我们生成 4 个通道
        img = torch.randn((1, 4, l), device=self.device)
        
        # 简化采样过程：为了不拖慢训练太多，我们可以只用 100 步 (DDIM) 或者完整跑完
        # 为了准确看到是否 collapse，建议完整跑完，或者至少跑 200 步
        # 这里复用标准的 DDPM 采样
        steps = self.diffusion.timesteps
        for i in reversed(range(0, steps)):
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

        # 2. 统计分布
        # 归一化回 [0, 1]
        img = (img.clamp(-1, 1) + 1) / 2
        grid = img[0].cpu().numpy() # [4, Length]
        
        # 设定阈值判断 Note
        threshold = 0.55
        notes = (grid > threshold).astype(int)
        
        # 按轨道求和
        counts = np.sum(notes, axis=1) # [Count_Col1, Count_Col2, ...]
        total_notes = np.sum(counts) + 1e-8 # 防止除以0
        
        # 3. 打印结果
        print("-" * 40)
        print(f"📊 分布统计 (Target SR: {sr.item():.1f})")
        print(f"   Total Notes: {int(total_notes)}")
        
        max_notes = np.max(counts)
        for k in range(4):
            count = counts[k]
            ratio = count / total_notes * 100
            # 简单的 ASCII 条形图
            bar_len = int((count / max_notes) * 20) if max_notes > 0 else 0
            bar = "█" * bar_len
            print(f"   Track {k+1}: {count:5d} ({ratio:5.1f}%) | {bar}")
            
        # 检查平衡性 (标准差)
        std_dev = np.std(counts / total_notes)
        if std_dev < 0.05:
            print("✅ 轨道分布均衡 (Good Balance)")
        elif std_dev > 0.15:
            print("⚠️ 轨道分布极度不均 (Possible Mode Collapse)")
            
        print("-" * 40)
        
        model.train() # 切回训练模式

# === 扩散模型辅助类 (补充 sample 需要的参数) ===
class DiffusionTrainer:
    def __init__(self, model, timesteps=1000):
        self.model = model
        self.timesteps = timesteps
        self.beta = torch.linspace(1e-4, 0.02, timesteps).to(DEVICE)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        # 预计算一些参数用于 q_sample
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat)

    def noise_images(self, x, t):
        """ 加噪过程 q(x_t | x_0) """
        # x 必须是 [-1, 1] 范围
        sqrt_alpha_hat = self.sqrt_alpha_hat[t][:, None, None]
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[t][:, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    # 在 DiffusionTrainer.train_step 中

    # 修改 DiffusionTrainer 的 train_step

    def train_step(self, x_start, mel, sr):
        t = self.sample_timesteps(x_start.shape[0])
        x_noisy, noise = self.noise_images(x_start, t)
        predicted_noise = self.model(x_noisy, mel, t, sr)
        
        # --- 回归最原始的 MSE Loss ---
        # 既然我们已经用了高斯热力图，target 本身就是平滑的
        # 不需要额外的 weight 来强制它学习 Note
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.timesteps, size=(n,), device=DEVICE)
# === 主训练循环 ===
def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    print(f"Using device: {DEVICE}")
    
    # 1. 数据准备 (开启 Augment!)
    dataset = ManiaDataset(DATA_DIR, sigma=2.0)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    # 2. 模型初始化
    model = ManiaUNet(in_channels=4, audio_channels=81, base_dim=64).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    diffusion = DiffusionTrainer(model, timesteps=TIMESTEPS)
    monitor = ValidationMonitor(diffusion, DEVICE)
    
    # === 抽取一个固定的验证样本 ===
    # 我们从数据集里取第一个样本，一直用它来观察模型变化
    sample_data = dataset[0] 
    # 增加 Batch 维度 [1, ...]
    fixed_mel = sample_data[0].unsqueeze(0).to(DEVICE) 
    fixed_sr = sample_data[2].unsqueeze(0).to(DEVICE)
    print(f"Validation Sample Loaded. Target SR: {fixed_sr.item()}")

    # 3. 开始训练
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        avg_loss = 0
        
        for mel, chart, sr in pbar:
            mel = mel.to(DEVICE)
            chart = chart.to(DEVICE)
            sr = sr.to(DEVICE)
            
            optimizer.zero_grad()
            loss = diffusion.train_step(chart, mel, sr)
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Average Loss: {avg_loss / len(dataloader):.6f}")
        
        # === 每个 Epoch 结束时进行监控 ===
        monitor.check(model, fixed_mel, fixed_sr, epoch+1)
        
        # 保存模型
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    main()