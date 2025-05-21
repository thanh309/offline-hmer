import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .pos_enc import WordPosEnc
from data.vocab import CROHMEVocab
from utils.hypothesis import Hypothesis

vocab = CROHMEVocab()
vocab_size = len(vocab)

class Decoder(nn.Module):
    def __init__(self, d_model, nhead, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            nn.LayerNorm(d_model)
        )
        self.pos_enc = WordPosEnc(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_mask, tgt):
        b, l = tgt.size()
        tgt_mask = torch.triu(torch.ones(l, l, device=tgt.device), diagonal=1).bool()
        tgt_pad_mask = tgt == vocab.PAD_IDX

        tgt_emb = self.word_embed(tgt)
        tgt_emb = self.pos_enc(tgt_emb)

        src = rearrange(src, 'b s d -> s b d')
        tgt_emb = rearrange(tgt_emb, 'b l d -> l b d')

        out = self.transformer(tgt_emb, src, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=src_mask)
        out = rearrange(out, 'l b d -> b l d')
        out = self.proj(out)
        return out
    def beam_search(self, src, mask, beam_size, max_len):
        assert src.size(0) == 1, "beam_search expects batch size 1"

        start_token = vocab.SOS_IDX
        stop_token = vocab.EOS_IDX

        hypotheses = torch.full((1, max_len + 1), vocab.PAD_IDX, dtype=torch.long, device=src.device)
        hypotheses[:, 0] = start_token
        hyp_scores = torch.zeros(1, device=src.device)
        completed_hyps = []

        t = 0
        while len(completed_hyps) < beam_size and t < max_len:
            hyp_num = hypotheses.size(0)
            exp_src = src.expand(hyp_num, -1, -1)
            exp_mask = mask.expand(hyp_num, -1)

            out = self.forward(exp_src, exp_mask, hypotheses)  # logits: [b, l, vocab_size]
            log_probs = torch.log_softmax(out[:, t, :], dim=-1)  # [b, vocab_size]

            new_hyp_scores = hyp_scores.unsqueeze(1) + log_probs  # [b, vocab_size]
            flat_scores = new_hyp_scores.view(-1)
            top_scores, top_indices = flat_scores.topk(beam_size - len(completed_hyps))

            prev_hyp_ids = top_indices // log_probs.size(1)
            next_token_ids = top_indices % log_probs.size(1)

            new_hypotheses = []
            new_hyp_scores = []

            for prev_hyp_id, next_token_id, score in zip(prev_hyp_ids, next_token_ids, top_scores):
                next_token_id = next_token_id.item()
                score = score.item()

                new_hyp = hypotheses[prev_hyp_id].clone()
                new_hyp[t + 1] = next_token_id

                if next_token_id == stop_token:
                    completed_hyps.append(Hypothesis(new_hyp[1:t+1], score, direction="l2r"))
                else:
                    new_hypotheses.append(new_hyp)
                    new_hyp_scores.append(score)

            if len(new_hypotheses) == 0:
                break

            hypotheses = torch.stack(new_hypotheses, dim=0)
            hyp_scores = torch.tensor(new_hyp_scores, device=src.device)

            t += 1

        if len(completed_hyps) == 0:
            completed_hyps.append(Hypothesis(hypotheses[0][1:], hyp_scores[0].item(), direction="l2r"))

        return completed_hyps
