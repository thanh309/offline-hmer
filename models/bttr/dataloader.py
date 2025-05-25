import os
from typing import List, Tuple
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from vocab import CROHMEVocab

class CROHMEDataset(Dataset):
    def __init__(self, root_dir: str):
        self.img_dir = os.path.join(root_dir, "img")
        caption_path = os.path.join(root_dir, "caption.txt")
        self.data = self._load_captions(caption_path)
        self.vocab = CROHMEVocab()
        self.to_tensor = ToTensor()

    def _load_captions(self, caption_path: str) -> List[Tuple[str, List[str]]]:
        with open(caption_path, "r", encoding="utf-8") as f:
            return [ (line.strip().split()[0], line.strip().split()[1:]) for line in f ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, formula = self.data[idx]
        img_path = os.path.join(self.img_dir, f"{img_name}.bmp")
        image = Image.open(img_path).convert("L")
        image_tensor = self.to_tensor(image)
        formula_indices = self.vocab.words2indices(formula)
        return img_name, image_tensor, formula_indices

def collate_fn(batch):
    fnames, images, formulas = zip(*batch)
    heights = [img.shape[1] for img in images]
    widths = [img.shape[2] for img in images]
    max_height, max_width = max(heights), max(widths)

    batch_size = len(images)
    imgs = torch.zeros(batch_size, 1, max_height, max_width)
    masks = torch.ones(batch_size, max_height, max_width, dtype=torch.bool)

    for i, img in enumerate(images):
        h, w = img.shape[1:]
        imgs[i, :, :h, :w] = img
        masks[i, :h, :w] = 0

    return fnames, imgs, masks, formulas
