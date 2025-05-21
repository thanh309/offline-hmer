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
