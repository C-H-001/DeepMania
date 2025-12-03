import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- 基础组件 ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.transform = nn.Conv1d(out_channels, out_channels, 3, padding=1)

        # 用于处理 Time + SR Embedding
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)  # Scale & Shift
        )

        self.bn = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        # x: [B, C, L]
        h = self.conv1(x)
        h = self.bn(h)
        h = self.act(h)
        h = self.transform(h)

        # 注入条件 (Time + SR)
        # FiLM 机制 (Feature-wise Linear Modulation)
        condition = self.mlp(t_emb).unsqueeze(-1)  # [B, C*2, 1]
        scale, shift = condition.chunk(2, dim=1)
        h = h * (scale + 1) + shift

        return h + self.conv1(x) if x.shape[1] == h.shape[1] else h


# --- 主模型 ---
class ManiaUNet(nn.Module):
    def __init__(self, in_channels=4, audio_channels=81, base_dim=64):
        super().__init__()

        self.input_dim = in_channels + audio_channels

        # Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_dim),
            nn.Linear(base_dim, base_dim * 4),
            nn.SiLU(),
            nn.Linear(base_dim * 4, base_dim * 4),
        )

        # SR Embedding (将难度标量映射为向量)
        self.sr_mlp = nn.Sequential(
            nn.Linear(1, base_dim),
            nn.SiLU(),
            nn.Linear(base_dim, base_dim * 4),
        )

        # Downsample Path
        self.down1 = Block1D(self.input_dim, base_dim, base_dim * 4)
        self.down2 = Block1D(base_dim, base_dim * 2, base_dim * 4)
        self.down3 = Block1D(base_dim * 2, base_dim * 4, base_dim * 4)

        self.pool = nn.MaxPool1d(2)

        # Bottleneck
        self.mid1 = Block1D(base_dim * 4, base_dim * 4, base_dim * 4)
        self.mid2 = Block1D(base_dim * 4, base_dim * 4, base_dim * 4)

        # Upsample Path
        self.up1 = Block1D(base_dim * 8, base_dim * 2, base_dim * 4)  # Concat: 4*base + 4*base
        self.up2 = Block1D(base_dim * 4, base_dim, base_dim * 4)  # Concat: 2*base + 2*base
        self.up3 = Block1D(base_dim * 2, base_dim,
                           base_dim * 4)  # Concat: base + base (extra conv needed usually, simplified here)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Output
        self.final_conv = nn.Conv1d(base_dim, in_channels, 1)

    def forward(self, x, mel, t, sr):
        """
        x: Noisy Chart [B, 4, L]
        mel: Audio Spectrogram [B, 80, L]
        t: Time Step [B]
        sr: Star Rating [B, 1]
        """
        # 1. 准备条件 Embeddings
        t_emb = self.time_mlp(t)
        sr_emb = self.sr_mlp(sr)
        # 将 Time 和 SR 相加作为总的 Context
        cond = t_emb + sr_emb

        # 2. Early Fusion: 将音频和带噪谱面在通道维度拼接
        # [B, 84, L]
        h = torch.cat([x, mel], dim=1)

        # 3. U-Net Forward
        # Down
        d1 = self.down1(h, cond)  # [64, L]
        d2 = self.down2(self.pool(d1), cond)  # [128, L/2]
        d3 = self.down3(self.pool(d2), cond)  # [256, L/4]

        # Mid
        m = self.mid1(self.pool(d3), cond)  # [256, L/8]
        m = self.mid2(m, cond)

        # Up (Concat with Skip Connections)
        u1 = self.upsample(m)
        u1 = torch.cat([u1, d3], dim=1)  # [512, L/4]
        u1 = self.up1(u1, cond)  # -> [128, L/4]

        u2 = self.upsample(u1)
        u2 = torch.cat([u2, d2], dim=1)  # [256, L/2]
        u2 = self.up2(u2, cond)  # -> [64, L/2]

        u3 = self.upsample(u2)
        u3 = torch.cat([u3, d1], dim=1)  # [128, L]
        u3 = self.up3(u3, cond)  # -> [64, L]

        return self.final_conv(u3)  # -> [4, L]