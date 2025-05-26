import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pos_enc import WordPosEnc
from vocab import CROHMEVocab
from hypothesis import Hypothesis

vocab = CROHMEVocab()
vocab_size = len(vocab)

class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, return_attention_weights=False):
        # Self attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention
        if return_attention_weights:
            tgt2, attn_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask, need_weights=True)
        else:
            tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        if return_attention_weights:
            return tgt, attn_weights
        else:
            return tgt

class CustomTransformerDecoder(nn.TransformerDecoder):
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, return_attention_weights=False):
        output = tgt
        intermediate = []
        attention_weights_list = []

        for mod in self.layers:
            if return_attention_weights:
                output, attn_weights = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, return_attention_weights=True)
                attention_weights_list.append(attn_weights)
            else:
                output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
            intermediate.append(output)

        if return_attention_weights:
            return output, attention_weights_list
        else:
            return output

class Decoder(nn.Module):
    def __init__(self, d_model, nhead, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            nn.LayerNorm(d_model)
        )
        self.pos_enc = WordPosEnc(d_model)
        # Use custom decoder layer
        decoder_layer = CustomTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer = CustomTransformerDecoder(decoder_layer, num_decoder_layers)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_mask, tgt, return_attention_weights=False):
        b, l = tgt.size()
        tgt_mask = torch.triu(torch.ones(l, l, device=tgt.device), diagonal=1).bool()
        tgt_pad_mask = tgt == vocab.PAD_IDX

        tgt_emb = self.word_embed(tgt)
        tgt_emb = self.pos_enc(tgt_emb)

        src = rearrange(src, 'b s d -> s b d')
        tgt_emb = rearrange(tgt_emb, 'b l d -> l b d')

        # Pass return_attention_weights to the transformer
        if return_attention_weights:
            out, attn_weights = self.transformer(tgt_emb, src, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=src_mask, return_attention_weights=True)
        else:
            out = self.transformer(tgt_emb, src, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=src_mask)

        out = rearrange(out, 'l b d -> b l d')
        out = self.proj(out)

        if return_attention_weights:
            return out, attn_weights
        else:
            return out
    def beam_search(self, src, mask, beam_size, max_len, alpha, vocab, feature_h, feature_w): # Accept h and w
        assert src.size(0) == 1, "beam_search expects batch size 1"

        start_token = vocab.SOS_IDX
        stop_token = vocab.EOS_IDX

        hypotheses = torch.full((1, max_len + 1), vocab.PAD_IDX, dtype=torch.long, device=src.device)
        hypotheses[:, 0] = start_token
        hyp_scores = torch.zeros(1, device=src.device)
        completed_hyps = []

        all_attentions = [] # List to store attention weights

        t = 0
        while len(completed_hyps) < beam_size and t < max_len:
            hyp_num = hypotheses.size(0)
            exp_src = src.expand(hyp_num, -1, -1)
            exp_mask = mask.expand(hyp_num, -1)

            # Call forward with return_attention_weights=True
            out, attn_weights = self.forward(exp_src, exp_mask, hypotheses, return_attention_weights=True)  # logits: [b, l, vocab_size]
            all_attentions.append(attn_weights) # Collect attention weights

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

        # Calculate scores explicitly
        scored_hyps = []
        for h in completed_hyps:
            # Use the alpha parameter
            score = h.score / (len(h) ** alpha)
            scored_hyps.append((score, h))

        # Find the best hypothesis based on the calculated scores
        best_hyp = max(scored_hyps, key=lambda x: x[0])[1] if scored_hyps else Hypothesis(torch.tensor([], device=src.device), 0.0)

        return best_hyp.seq, all_attentions, feature_h, feature_w # Return sequence, attentions, h, and w
