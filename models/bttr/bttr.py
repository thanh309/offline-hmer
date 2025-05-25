import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class BTTR(nn.Module):
    def __init__(self, d_model, growth_rate, num_layers, nhead, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.encoder = Encoder(d_model, growth_rate, num_layers)
        self.decoder = Decoder(d_model, nhead, num_decoder_layers, dim_feedforward, dropout)

    def forward(self, img, img_mask, tgt):
        features, mask = self.encoder(img, img_mask)
        features = torch.cat([features, features], dim=0)
        mask = torch.cat([mask, mask], dim=0)
        return self.decoder(features, mask, tgt)

    def beam_search(self, img, img_mask, beam_size, max_len):
        features, mask = self.encoder(img, img_mask)
        return self.decoder.beam_search(features, mask, beam_size, max_len)
