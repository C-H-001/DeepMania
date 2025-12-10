import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ================= 基础组件 =================

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

class Downsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim
        self.conv = nn.Conv1d(dim, dim_out, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim
        self.conv = nn.ConvTranspose1d(dim, dim_out, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

# ================= 核心模块：ResNet Block =================
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim is not None else None

        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim_out, 3, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, dim_out),
            nn.SiLU(),
            nn.Conv1d(dim_out, dim_out, 3, padding=1)
        )

        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)

        if self.mlp is not None:
            condition = self.mlp(time_emb).unsqueeze(-1)
            scale, shift = condition.chunk(2, dim=1)
            h = h * (1 + scale) + shift

        h = self.block2(h)
        return h + self.res_conv(x)

# ================= 核心模块：Attention Block =================
class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = nn.GroupNorm(32, dim)
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, l = x.shape
        normed_x = self.norm(x)
        qkv = self.to_qkv(normed_x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, l), qkv)

        q = q * self.scale
        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h d j -> b h d i', attn, v)
        
        out = out.reshape(b, -1, l)
        return self.to_out(out) + x

# ================= 主模型：ManiaDiffuserV2 =================
class ManiaDiffuserV2(nn.Module):
    def __init__(
        self, 
        in_channels=4, 
        audio_channels=81, 
        base_dim=128,
        dim_mults=(1, 2, 4, 8) 
    ):
        super().__init__()
        
        self.input_dim = in_channels + audio_channels
        
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.sr_mlp = nn.Sequential(
            nn.Linear(1, base_dim),
            nn.GELU(),
            nn.Linear(base_dim, time_dim),
        )

        self.init_conv = nn.Conv1d(self.input_dim, base_dim, 3, padding=1)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        
        dims = [base_dim, *map(lambda m: base_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:])) 
        
        # --- Downsample Path ---
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                AttentionBlock(dim_in) if ind >= 2 else nn.Identity(),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        
        # --- Bottleneck ---
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = AttentionBlock(mid_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        # --- Upsample Path ---
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            
            self.ups.append(nn.ModuleList([
                # Block 1: 负责融合 skip connection
                # 输入: dim_out (来自深层) + dim_in (来自 skip)
                # 输出: dim_in (恢复到当前层维度)
                ResnetBlock(dim_out + dim_in, dim_in, time_emb_dim=time_dim),
                
                # Block 2: 负责特征提炼
                # 输入: dim_in
                # 输出: dim_in
                # 【修正点】这里之前写成了 dim_in + dim_in，是错的
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim), 
                
                AttentionBlock(dim_in) if (len(in_out) - ind - 1) >= 2 else nn.Identity(),
                Upsample(dim_in, dim_in) if not is_last else nn.Conv1d(dim_in, dim_in, 3, padding=1)
            ]))

        self.final_res_block = ResnetBlock(base_dim, base_dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv1d(base_dim, in_channels, 1)

    def forward(self, x, mel, t, sr):
        t_emb = self.time_mlp(t)
        sr_emb = self.sr_mlp(sr)
        cond = t_emb + sr_emb

        h = self.init_conv(torch.cat([x, mel], dim=1))
        
        hiddens = []
        for block1, block2, attn, downsample in self.downs:
            h = block1(h, cond)
            h = block2(h, cond)
            h = attn(h)
            hiddens.append(h)
            h = downsample(h)

        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)

        for block1, block2, attn, upsample in self.ups:
            skip = hiddens.pop()
            h = torch.cat((h, skip), dim=1)
            
            h = block1(h, cond)
            h = block2(h, cond)
            h = attn(h)
            h = upsample(h)

        h = self.final_res_block(h, cond)
        h = self.final_conv(h)
        return h