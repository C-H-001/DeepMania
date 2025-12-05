import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 组件定义
# ==========================================

class SinusoidalPosEmb(nn.Module):
    """
    正弦位置编码，用于将时间步 t 映射为向量
    """
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

class SelfAttention1D(nn.Module):
    """
    轻量级 1D 自注意力机制
    用于在 Bottleneck 层捕捉全局上下文（比如整段的节奏型）
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.norm = nn.GroupNorm(1, channels)
        
        # Q, K, V 投影
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        # x: [Batch, Channels, Length]
        B, C, L = x.shape
        
        # 转换维度以适应 Linear 层: [B, L, C]
        x_in = x.permute(0, 2, 1)
        x_norm = self.norm(x).permute(0, 2, 1)
        
        # 计算 QKV
        qkv = self.qkv(x_norm).reshape(B, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        
        # Scaled Dot-Product Attention
        # attn: [B, Heads, L, L]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 加权求和
        x_out = (attn @ v).transpose(1, 2).reshape(B, L, C)
        
        # 最终投影 + 残差连接
        x_out = self.proj(x_out)
        return (x_out + x_in).permute(0, 2, 1)

class Block1D(nn.Module):
    """
    ResNet 风格的卷积块，支持时间步和SR条件的注入
    """
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        
        # 条件注入层 (Scale & Shift)
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels * 2) 
        )
        
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.bn2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        # 如果输入输出通道不一致，用于残差连接的 1x1 卷积
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        # x: [B, C, L]
        # cond: [B, Cond_Dim]
        
        # Res path
        res = self.residual_conv(x)
        
        # Layer 1
        h = self.conv1(x)
        h = self.bn1(h)
        
        # 注入条件 (FiLM: Feature-wise Linear Modulation)
        # 将 Time+SR 的向量映射为 Scale 和 Shift，作用于特征图
        cond_scale, cond_shift = self.cond_mlp(cond).unsqueeze(-1).chunk(2, dim=1)
        h = h * (1 + cond_scale) + cond_shift
        
        h = self.act(h)
        
        # Layer 2
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.act(h)
        
        return h + res

# ==========================================
# 主模型架构
# ==========================================

class ManiaUNet(nn.Module):
    def __init__(self, in_channels=4, audio_channels=81, base_dim=64):
        super().__init__()
        
        # 输入维度 = 谱面通道(4) + 音频通道(81: 80Mel+1Onset)
        self.input_dim = in_channels + audio_channels
        
        # 1. Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_dim),
            nn.Linear(base_dim, base_dim * 4),
            nn.SiLU(),
            nn.Linear(base_dim * 4, base_dim * 4),
        )
        
        # 2. SR (Star Rating) Embedding
        self.sr_mlp = nn.Sequential(
            nn.Linear(1, base_dim),
            nn.SiLU(),
            nn.Linear(base_dim, base_dim * 4),
        )

        # 3. Encoder (Downsampling Path)
        # Input -> 64
        self.down1 = Block1D(self.input_dim, base_dim, base_dim * 4)
        # 64 -> 128
        self.down2 = Block1D(base_dim, base_dim * 2, base_dim * 4)
        # 128 -> 256
        self.down3 = Block1D(base_dim * 2, base_dim * 4, base_dim * 4)
        
        self.pool = nn.MaxPool1d(2)

        # 4. Bottleneck (Mid Path) with Attention
        self.mid1 = Block1D(base_dim * 4, base_dim * 4, base_dim * 4)
        
        # [关键升级] 在中间层加入 Self-Attention
        self.attn = SelfAttention1D(base_dim * 4, num_heads=4)
        
        self.mid2 = Block1D(base_dim * 4, base_dim * 4, base_dim * 4)

        # 5. Decoder (Upsampling Path)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Concat(256 + 256) -> 128
        self.up1 = Block1D(base_dim * 8, base_dim * 2, base_dim * 4) 
        # Concat(128 + 128) -> 64
        self.up2 = Block1D(base_dim * 4, base_dim, base_dim * 4)     
        # Concat(64 + 64) -> 64
        self.up3 = Block1D(base_dim * 2, base_dim, base_dim * 4)     
        
        # 6. Final Output
        self.final_conv = nn.Conv1d(base_dim, in_channels, 1)

    def forward(self, x, mel, t, sr):
        """
        x: Noisy Chart [B, 4, L]
        mel: Audio Features [B, 81, L]
        t: Time Step [B]
        sr: Star Rating [B, 1]
        """
        # --- 条件处理 ---
        t_emb = self.time_mlp(t)
        sr_emb = self.sr_mlp(sr)
        # 将 Time 和 SR 融合作为全局条件
        cond = t_emb + sr_emb
        
        # --- Early Fusion ---
        # 将音频特征直接拼接到输入层
        # h shape: [B, 85, L]
        h = torch.cat([x, mel], dim=1)
        
        # --- Downsample ---
        d1 = self.down1(h, cond) # -> [B, 64, L]
        
        x_pool = self.pool(d1)
        d2 = self.down2(x_pool, cond) # -> [B, 128, L/2]
        
        x_pool = self.pool(d2)
        d3 = self.down3(x_pool, cond) # -> [B, 256, L/4]
        
        # --- Bottleneck ---
        x_pool = self.pool(d3)
        m = self.mid1(x_pool, cond) # -> [B, 256, L/8]
        
        # 应用 Attention (让模型理解全局节奏结构)
        m = self.attn(m)
        
        m = self.mid2(m, cond)
        
        # --- Upsample ---
        u1 = self.upsample(m)
        # Skip Connection from d3
        u1 = torch.cat([u1, d3], dim=1) 
        u1 = self.up1(u1, cond)         # -> [B, 128, L/4]
        
        u2 = self.upsample(u1)
        # Skip Connection from d2
        u2 = torch.cat([u2, d2], dim=1) 
        u2 = self.up2(u2, cond)         # -> [B, 64, L/2]
        
        u3 = self.upsample(u2)
        # Skip Connection from d1
        u3 = torch.cat([u3, d1], dim=1) 
        u3 = self.up3(u3, cond)         # -> [B, 64, L]
        
        # --- Final Projection ---
        return self.final_conv(u3) # -> [B, 4, L]