import os
from functools import lru_cache
from typing import Dict, List

@lru_cache()
def default_dict():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "dictionary.txt")

class CROHMEVocab:
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2

    def __init__(self, dict_path: str = default_dict()) -> None:
        self.word2idx = {"<pad>": self.PAD_IDX, "<sos>": self.SOS_IDX, "<eos>": self.EOS_IDX}
        with open(dict_path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                self.word2idx[w] = len(self.word2idx)
        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}

    def words2indices(self, words: List[str]) -> List[int]:
        return [self.word2idx[w] for w in words]

    def indices2words(self, id_list: List[int]) -> List[str]:
        return [self.idx2word[i] for i in id_list]

    def indices2label(self, id_list: List[int]) -> str:
        return " ".join(self.indices2words(id_list))

    def __len__(self):
        return len(self.word2idx)
