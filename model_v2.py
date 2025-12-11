import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ... (SinusoidalPosEmb, Downsample, Upsample, ResnetBlock, AttentionBlock 保持不变) ...
# 注意：Upsample 建议改回 ConvTranspose1d 或者我之前给的插值版本，这里为了省空间省略基础模块代码

class ManiaDiffuserV2(nn.Module):
    def __init__(
        self, 
        in_channels=8,       # 4 Head + 4 Body
        audio_channels=83,   # 80 Mel + 3 Onset
        base_dim=128,
        dim_mults=(1, 2, 4, 8) 
    ):
        super().__init__()
        
        self.input_dim = in_channels + audio_channels
        time_dim = base_dim * 4
        
        # Embeddings
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
        self.ln_mlp = nn.Sequential(
            nn.Linear(1, base_dim),
            nn.GELU(),
            nn.Linear(base_dim, time_dim),
        )
        
        # 【核心改动】Null Token for Auto Mode
        self.null_ln_emb = nn.Parameter(torch.randn(time_dim))

        self.init_conv = nn.Conv1d(self.input_dim, base_dim, 3, padding=1)

        # U-Net Structure
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        
        dims = [base_dim, *map(lambda m: base_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:])) 
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                AttentionBlock(dim_in) if ind >= 2 else nn.Identity(),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = AttentionBlock(mid_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_in, time_emb_dim=time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                AttentionBlock(dim_in) if (len(in_out) - ind - 1) >= 2 else nn.Identity(),
                Upsample(dim_in, dim_in) if not is_last else nn.Conv1d(dim_in, dim_in, 3, padding=1)
            ]))

        self.final_res_block = ResnetBlock(base_dim, base_dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv1d(base_dim, in_channels, 1)

    def forward(self, x, mel, t, sr, ln_ratio):
        t_emb = self.time_mlp(t)
        sr_emb = self.sr_mlp(sr)
        
        # 【核心改动】CFG Logic
        # 如果 ln_ratio < 0，使用 null_ln_emb
        ln_emb_calc = self.ln_mlp(ln_ratio)
        mask = (ln_ratio < 0).float() # [B, 1]
        mask = mask.view(-1, 1).expand_as(ln_emb_calc)
        
        batch_null = self.null_ln_emb.unsqueeze(0).expand_as(ln_emb_calc)
        final_ln = ln_emb_calc * (1 - mask) + batch_null * mask
        
        cond = t_emb + sr_emb + final_ln

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