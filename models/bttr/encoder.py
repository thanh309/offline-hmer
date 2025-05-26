import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pos_enc import ImgPosEnc

class _Bottleneck(nn.Module):
    def __init__(self, in_ch, growth_rate, use_dropout):
        super().__init__()
        inter_ch = 4 * growth_rate
        self.conv1 = nn.Conv2d(in_ch, inter_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_ch)
        self.conv2 = nn.Conv2d(inter_ch, growth_rate, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate)
        self.dropout = nn.Dropout(0.2) if use_dropout else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        return torch.cat([x, out], dim=1)

class _Transition(nn.Module):
    def __init__(self, in_ch, out_ch, use_dropout):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout(0.2) if use_dropout else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out
    
class _SingleLayer(nn.Module):
    def __init__(self, in_ch, growth_rate, use_dropout):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, growth_rate, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(growth_rate)
        self.dropout = nn.Dropout(0.2) if use_dropout else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.dropout(out)
        return torch.cat([x, out], dim=1)


class DenseNet(nn.Module):
    def __init__(self, growth_rate, num_layers, reduction=0.5, bottleneck=True, use_dropout=True):
        super().__init__()
        n_ch = 2 * growth_rate
        self.conv1 = nn.Conv2d(1, n_ch, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(n_ch)

        self.dense1 = self._make_dense(n_ch, growth_rate, num_layers, bottleneck, use_dropout)
        n_ch += num_layers * growth_rate
        out_ch = int(n_ch * reduction)
        self.trans1 = _Transition(n_ch, out_ch, use_dropout)

        n_ch = out_ch
        self.dense2 = self._make_dense(n_ch, growth_rate, num_layers, bottleneck, use_dropout)
        n_ch += num_layers * growth_rate
        out_ch = int(n_ch * reduction)
        self.trans2 = _Transition(n_ch, out_ch, use_dropout)

        n_ch = out_ch
        self.dense3 = self._make_dense(n_ch, growth_rate, num_layers, bottleneck, use_dropout)
        n_ch += num_layers * growth_rate

        self.post_norm = nn.BatchNorm2d(n_ch)
        self.out_channels = n_ch

    def _make_dense(self, in_ch, growth_rate, num_layers, bottleneck, use_dropout):
        layers = []
        for _ in range(num_layers):
            layers.append(_Bottleneck(in_ch, growth_rate, use_dropout) if bottleneck else _SingleLayer(in_ch, growth_rate, use_dropout))
            in_ch += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, mask):
        out = self.conv1(x)
        out = self.norm1(out)
        out_mask = mask[:, ::2, ::2]

        out = F.relu(out)
        out = F.max_pool2d(out, 2, ceil_mode=True)
        out_mask = out_mask[:, ::2, ::2]

        out = self.trans1(self.dense1(out))
        out_mask = out_mask[:, ::2, ::2]
        out = self.trans2(self.dense2(out))
        out_mask = out_mask[:, ::2, ::2]
        out = self.dense3(out)
        out = self.post_norm(out)
        return out, out_mask

class Encoder(nn.Module):
    def __init__(self, d_model, growth_rate, num_layers):
        super().__init__()
        self.densenet = DenseNet(growth_rate, num_layers)
        self.feature_proj = nn.Conv2d(self.densenet.out_channels, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc = ImgPosEnc(d_model, normalize=True)

    def forward(self, img, mask):
        features, mask = self.densenet(img, mask)
        features = self.feature_proj(features)
        # Get h and w before flattening
        b, d, h, w = features.size()
        features = rearrange(features, 'b d h w -> b h w d')
        features = self.norm(features)
        features = self.pos_enc(features, mask)
        features = rearrange(features, 'b h w d -> b (h w) d')
        mask = rearrange(mask, 'b h w -> b (h w)')
        return features, mask, h, w # Return h and w
