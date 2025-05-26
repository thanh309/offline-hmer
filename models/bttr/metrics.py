import torch
from torchmetrics import Metric
import editdistance

class ExpRateRecorder(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, indices_hat, indices):
        dist = editdistance.eval(indices_hat, indices)
        if dist == 0:
            self.correct += 1
        self.total += 1

    def compute(self):
        return (self.correct / self.total).item() if self.total > 0 else 0.0

class CROHMERecorder(Metric):
    def __init__(self, vocab, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.vocab = vocab

        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("exp_match", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("error1_match", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("error2_match", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("stru_match", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def strip_symbols(self, seq):
        structure_tokens = {'\\frac', '\\sqrt', '\\int', '\\sum', '(', ')', '[', ']', '{', '}', '=', '+', '-', '\\rightarrow', '\\cdot'}
        return [t for t in seq if self.vocab.idx2word.get(t, '') in structure_tokens]

    def update(self, pred_indices, gt_indices):
        # Exact match and tolerance errors
        dist = editdistance.eval(pred_indices, gt_indices)

        if dist == 0:
            self.exp_match += 1
        if dist <= 1:
            self.error1_match += 1
        if dist <= 2:
            self.error2_match += 1
        if dist <= 3:
            self.error3_match += 1

        # Structure comparison
        pred_struct = self.strip_symbols(pred_indices)
        gt_struct = self.strip_symbols(gt_indices)
        stru_dist = editdistance.eval(pred_struct, gt_struct)
        if stru_dist == 0:
            self.stru_match += 1

        self.total += 1

    def compute(self):
        if self.total == 0:
            return {"ExpRate": 0.0, "1error": 0.0, "2error": 0.0, "StruRate": 0.0}

        return {
            "ExpRate": (self.exp_match / self.total).item(),
            "1error": (self.error1_match / self.total).item(),
            "2error": (self.error2_match / self.total).item(),
            "StruRate": (self.stru_match / self.total).item()
        }
