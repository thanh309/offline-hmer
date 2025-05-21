import torch
import torch.nn.functional as F
from einops import rearrange

from data.vocab import CROHMEVocab

vocab = CROHMEVocab()

def ce_loss(output_hat: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    flat_hat = rearrange(output_hat, "b l e -> (b l) e")
    flat = rearrange(output, "b l -> (b l)")
    return F.cross_entropy(flat_hat, flat, ignore_index=vocab.PAD_IDX)

def to_tgt_output(tokens, direction, device):
    assert direction in {"l2r", "r2l"}

    tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]
    if direction == "l2r":
        start_w, stop_w = vocab.SOS_IDX, vocab.EOS_IDX
    else:
        tokens = [torch.flip(t, dims=[0]) for t in tokens]
        start_w, stop_w = vocab.EOS_IDX, vocab.SOS_IDX

    batch_size = len(tokens)
    max_len = max(len(t) for t in tokens)
    tgt = torch.full((batch_size, max_len + 1), vocab.PAD_IDX, dtype=torch.long, device=device)
    out = torch.full((batch_size, max_len + 1), vocab.PAD_IDX, dtype=torch.long, device=device)

    for i, t in enumerate(tokens):
        tgt[i, 0] = start_w
        tgt[i, 1:1+len(t)] = t
        out[i, :len(t)] = t
        out[i, len(t)] = stop_w

    return tgt, out

def to_bi_tgt_out(tokens, device):
    l2r_tgt, l2r_out = to_tgt_output(tokens, "l2r", device)
    r2l_tgt, r2l_out = to_tgt_output(tokens, "r2l", device)
    tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
    out = torch.cat((l2r_out, r2l_out), dim=0)
    return tgt, out
