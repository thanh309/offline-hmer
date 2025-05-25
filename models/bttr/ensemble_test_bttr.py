import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import CROHMEDataset, collate_fn
from bttr import BTTR
from metrics import ExpRateRecorder
from beam_search import ensemble_beam_search_batch
from vocab import CROHMEVocab

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # Config
    checkpoint_paths = [
        # "checkpoints/epoch_60.pth"
    ]
    data_root = "resources/CROHME/test"
    batch_size = 4
    beam_size = 10
    max_len = 200
    alpha = 1.0

    # Data
    vocab = CROHMEVocab()
    test_dataset = CROHMEDataset(data_root)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Load Models
    models = []
    for path in checkpoint_paths:
        model = BTTR(
            d_model=256,
            growth_rate=16,
            num_layers=3,
            nhead=8,
            num_decoder_layers=3,
            dim_feedforward=1024,
            dropout=0.1
        ).to(DEVICE)
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    # Metrics
    recorder = ExpRateRecorder()

    # Inference Loop
    os.makedirs("results_ensemble", exist_ok=True)
    with torch.no_grad():
        for fnames, imgs, masks, formulas in tqdm(test_loader, desc="Ensemble Testing"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            hypotheses_batch = ensemble_beam_search_batch(models, imgs, masks, beam_size, max_len, alpha, vocab)

            for fname, hyp, gt_formula in zip(fnames, hypotheses_batch, formulas):
                pred_latex = vocab.indices2label(hyp)
                gt_indices = vocab.words2indices(gt_formula)

                recorder.update(hyp, gt_indices)

                # Save result file
                with open(f"results_ensemble/{fname}.txt", "w", encoding="utf-8") as f:
                    f.write(f"%{fname}\n${pred_latex}$")

    # Final Metrics
    exprate = recorder.compute()
    print(f"Ensemble Expression Recognition Rate: {exprate:.4f}")

if __name__ == "__main__":
    main()
