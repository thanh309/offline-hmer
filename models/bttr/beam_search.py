import torch
from hypothesis import Hypothesis

def beam_search_batch(model, imgs, masks, beam_size, max_len, alpha, vocab):
    batch_size = imgs.size(0)
    results = []
    all_batch_attentions = []
    all_feature_h = []
    all_feature_w = []
    for i in range(batch_size):
        img, mask = imgs[i].unsqueeze(0), masks[i].unsqueeze(0)
        seq, attentions, feature_h, feature_w = model.beam_search(img, mask, beam_size, max_len, alpha, vocab)
        results.append(seq)
        all_batch_attentions.append(attentions)
        all_feature_h.append(feature_h)
        all_feature_w.append(feature_w)
    return results, all_batch_attentions, all_feature_h, all_feature_w

def ensemble_beam_search_batch(models, imgs, masks, beam_size, max_len, alpha, vocab):
    batch_size = imgs.size(0)
    results = []
    all_batch_attentions = []
    all_feature_h = []
    all_feature_w = []
    for i in range(batch_size):
        img, mask = imgs[i].unsqueeze(0), masks[i].unsqueeze(0)
        all_hyps = []
        batch_attentions = []
        batch_feature_h = []
        batch_feature_w = []
        for model in models:
            seq, attentions, feature_h, feature_w = model.beam_search(img, mask, beam_size, max_len, alpha, vocab)
            all_hyps.append(Hypothesis(seq, 0.0, direction="l2r"))  # Score will be recomputed
            batch_attentions.append(attentions)
            batch_feature_h.append(feature_h)
            batch_feature_w.append(feature_w)
        best = max(all_hyps, key=lambda h: h.score / (len(h) ** alpha))
        results.append(best.seq)
        all_batch_attentions.append(batch_attentions[all_hyps.index(best)])
        all_feature_h.append(batch_feature_h[all_hyps.index(best)])
        all_feature_w.append(batch_feature_w[all_hyps.index(best)])
    return results, all_batch_attentions, all_feature_h, all_feature_w