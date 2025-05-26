import math
import torch
import torch.nn as nn
from einops import rearrange, repeat

class WordPosEnc(nn.Module):
    def __init__(self, d_model, max_len=500, temperature=10000.0):
        super().__init__()
        pos = torch.arange(0, max_len).float()
        dim_t = torch.arange(0, d_model, 2).float()
        div_term = 1.0 / (temperature ** (dim_t / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos[:, None] * div_term)
        pe[:, 1::2] = torch.cos(pos[:, None] * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        _, seq_len, _ = x.size()
        return x + self.pe[:seq_len, :].unsqueeze(0) # (batch_size, seq_len, d_model)

class ImgPosEnc(nn.Module):
    def __init__(self, d_model, temperature=10000.0, normalize=False):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D encoding"
        self.half_d_model = d_model // 2
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, x, mask):
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi

        dim_t = torch.arange(self.half_d_model // 2, dtype=torch.float32, device=x.device) #Calculate inversed frequency
        inv_freq = 1.0 / (self.temperature ** (dim_t / (self.half_d_model // 2)))

        pos_x = torch.einsum('b h w, d -> b h w d', x_embed, inv_freq)
        pos_y = torch.einsum('b h w, d -> b h w d', y_embed, inv_freq)

        pos_x = torch.cat([pos_x.sin(), pos_x.cos()], dim=-1)
        pos_y = torch.cat([pos_y.sin(), pos_y.cos()], dim=-1)
        pos = torch.cat([pos_x, pos_y], dim=-1)  # final shape: [b, h, w, d_model]

        assert pos.shape[-1] == x.shape[-1], f"PosEnc shape mismatch: {pos.shape[-1]} != {x.shape[-1]}"
        return x + pos
