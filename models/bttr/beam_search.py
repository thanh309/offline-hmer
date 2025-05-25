import torch
from hypothesis import Hypothesis

def beam_search_batch(model, imgs, masks, beam_size, max_len, alpha, vocab):
    batch_size = imgs.size(0)
    results = []
    for i in range(batch_size):
        img, mask = imgs[i].unsqueeze(0), masks[i].unsqueeze(0)
        hyps = model.beam_search(img, mask, beam_size, max_len)
        best = max(hyps, key=lambda h: h.score / (len(h) ** alpha))
        results.append(best.seq)
    return results

def ensemble_beam_search_batch(models, imgs, masks, beam_size, max_len, alpha, vocab):
    batch_size = imgs.size(0)
    results = []
    for i in range(batch_size):
        img, mask = imgs[i].unsqueeze(0), masks[i].unsqueeze(0)
        all_hyps = []
        for model in models:
            hyps = model.beam_search(img, mask, beam_size, max_len)
            all_hyps.extend(hyps)
        best = max(all_hyps, key=lambda h: h.score / (len(h) ** alpha))
        results.append(best.seq)
    return results
