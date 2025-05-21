import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
# import wandb

from data.dataloader import CROHMEDataset, collate_fn
from models.bttr.bttr import BTTR
from utils.losses import ce_loss, to_bi_tgt_out
from utils.metrics import ExpRateRecorder
from data.vocab import CROHMEVocab
from utils.beam_search import beam_search_batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = CROHMEVocab()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# wandb.login(
#     key = os.getenv("WANDB_KEY"),
# )
# wandb.init(
#     project = "BTTR_offline_hmer"
# )


def main():

    # wandb.init(project=CONFIG["project"], config=CONFIG)

    # Configs
    train_root = "resources/CROHME/train"
    val_root = "resources/CROHME/val"
    batch_size = 8
    num_epochs = 50
    learning_rate = 1.0
    patience = 10
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Data Loaders
    train_dataset = CROHMEDataset(train_root)
    val_dataset = CROHMEDataset(val_root)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Model
    model = BTTR(
        d_model=256,
        growth_rate=16,
        num_layers=3,
        nhead=8,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1
    ).to(DEVICE)

    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, eps=1e-6, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=patience)

    exp_rate_recorder = ExpRateRecorder()

    # Training Loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0

        for fnames, imgs, masks, formulas in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            tgt, out = to_bi_tgt_out(formulas, DEVICE)

            optimizer.zero_grad()
            out_hat = model(imgs, masks, tgt)
            loss = ce_loss(out_hat, out)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        exp_rate_recorder.reset()

        with torch.no_grad():
            for fnames, imgs, masks, formulas in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                tgt, out = to_bi_tgt_out(formulas, DEVICE)

                out_hat = model(imgs, masks, tgt)
                loss = ce_loss(out_hat, out)
                val_loss += loss.item()

                # Optional: Add beam search for validation exprate if needed
                preds = beam_search_batch(model, imgs, masks, beam_size=10, max_len=200, alpha=1.0, vocab=vocab)
                gt_indices = vocab.words2indices(formulas[0])
                exp_rate_recorder.update(preds[0], gt_indices)

        avg_val_loss = val_loss / len(val_loader)
        val_exprate = exp_rate_recorder.compute()
        print(f"Epoch {epoch}: Val Loss = {avg_val_loss:.4f} | Val Exprate = {val_exprate:.4f}")

        torch.cuda.empty_cache()

        # Scheduler step based on ExpRate (set to dummy for now)
        scheduler.step(val_exprate)

        # wandb.log({
        #     "epoch": epoch,
        #     "train_loss": avg_train_loss,
        #     "val_loss": avg_val_loss
        # })
        wandb.log({"Epoch": epoch, "Train loss": avg_train_loss, "Valid loss": avg_val_loss, "Val exprate": val_exprate})
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch}.pth"))
    # wandb.finish()

if __name__ == "__main__":
    main()
