import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import CROHMEDataset, collate_fn
from bttr import BTTR
from metrics import ExpRateRecorder
from beam_search import ensemble_beam_search_batch
from vocab import CROHMEVocab
from beam_search import beam_search_batch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

def main():
    # Config
    checkpoint_path = "checkpoints/bttr_best.pth"
    data_root = "resources/CROHME/test"
    output_dir = "./inference_results"
    beam_size = 10
    max_len = 200

    os.makedirs(output_dir, exist_ok=True)

    # Load vocab
    vocab = CROHMEVocab()

    # Load dataset
    test_dataset = CROHMEDataset(data_root)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Load model
    model = BTTR(d_model=256, growth_rate=16, num_layers=3, nhead=8, num_decoder_layers=3, dim_feedforward=1024, dropout=0.3).to("cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Inference
    with torch.no_grad():
        for i, (fnames, imgs, masks, formulas) in enumerate(tqdm(test_loader, desc="Inference")):
            if i >= 5:
                break
            imgs, masks = imgs.to("cpu"), masks.to("cpu")

            hypotheses, batch_attention_weights, batch_feature_h, batch_feature_w = beam_search_batch(model, imgs, masks, beam_size=beam_size, max_len=max_len, alpha=1.0, vocab=vocab)
            pred_seq = hypotheses[0]
            sample_attention_weights = batch_attention_weights[0]
            feature_h = batch_feature_h[0]
            feature_w = batch_feature_w[0]

            gt_seq = formulas[0]

            pred_latex = vocab.indices2label(pred_seq)
            with open(os.path.join(output_dir, f"{fnames[0]}.txt"), "w", encoding="utf-8") as f:
                f.write(f"%{fnames[0]}\n${pred_latex}$")

            # Load original image
            original_image_path = os.path.join(data_root, "img", f"{fnames[0]}.bmp")
            original_image = Image.open(original_image_path).convert("RGB")

            # Placeholder for best crop
            best_crop = (0, 0, original_image.size[0], original_image.size[1])

            # Process attention weights for visualization
            processed_attention_weights = []
            num_layers = len(sample_attention_weights[0])
            num_heads = sample_attention_weights[0][0].shape[0]

            for t in range(len(pred_latex.split())):
                token_attention_weights = [sample_attention_weights[t][layer_idx] for layer_idx in range(num_layers)]
                concatenated_attention = torch.cat(token_attention_weights, dim=0)  # Shape: (num_layers * num_heads, batch_size, 1, source_len)
                averaged_attention = torch.mean(concatenated_attention, dim=(0, 1))  # Average across layers/heads and batch
                averaged_attention = averaged_attention.squeeze()  # Shape: (source_len,)
                processed_attention_weights.append(averaged_attention)

            # Visualize attention maps
            visualize_attention_maps(original_image, processed_attention_weights, pred_latex.split(), best_crop, feature_h, feature_w)

def visualize_attention_maps(orig_image: Image, alphas, latex_tokens, best_crop, feature_h, feature_w, max_cols=4):
    '''
    Visualize attention maps over the original (unpadded) image
    '''
    orig_image = orig_image.crop(best_crop)
    orig_w, orig_h = orig_image.size

    num_tokens = len(latex_tokens)
    num_cols = min(max_cols, num_tokens)
    num_rows = int(np.ceil(num_tokens / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3 * orig_h / orig_w))
    axes = np.array(axes).reshape(-1)

    for i, (token, alpha) in enumerate(zip(latex_tokens, alphas)):
        ax = axes[i]

        # Verify alpha shape
        expected_source_len = feature_h * feature_w
        if alpha.shape[0] != expected_source_len:
            raise ValueError(f"Attention weight shape {alpha.shape} does not match expected source_len {expected_source_len}")

        # Reshape to (feature_h, feature_w)
        alpha = alpha.view(feature_h, feature_w)

        # Resize to (orig_h, orig_w) for overlay
        alpha = cv2.resize(alpha.cpu().numpy(), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        ax.imshow(orig_image)
        ax.imshow(alpha, cmap='jet', alpha=0.4)
        ax.set_title(f'{token}', fontsize=10 * orig_h / orig_w)
        ax.axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig('attention_maps_bttr.png', bbox_inches='tight', dpi=150)
    plt.close()

if __name__ == "__main__":
    main()