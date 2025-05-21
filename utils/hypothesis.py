from typing import List
import torch

class Hypothesis:
    def __init__(self, seq_tensor: torch.Tensor, score: float, direction: str):
        assert direction in {"l2r", "r2l"}
        raw_seq = seq_tensor.tolist()
        self.seq = raw_seq[::-1] if direction == "r2l" else raw_seq
        self.score = score

    def __len__(self):
        return len(self.seq) if self.seq else 1

    def __str__(self):
        return f"seq: {self.seq}, score: {self.score}"
