import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
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

device = "cuda" if torch.cuda.is_available() else "cpu"


# def main():
#     # Config
#     checkpoint_path = "checkpoints/bttr_best.pth"
#     data_root = "resources/CROHME/test"
#     output_dir = "./inference_results"
#     beam_size = 10
#     max_len = 200

#     os.makedirs(output_dir, exist_ok=True)

#     # Load vocab
#     vocab = CROHMEVocab()

#     # Load dataset
#     test_dataset = CROHMEDataset(data_root)
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

#     # Load model
#     model = BTTR(d_model=256, growth_rate=16, num_layers=3, nhead=8, num_decoder_layers=3, dim_feedforward=1024, dropout=0.3).to(device)
#     checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
#     model.load_state_dict(checkpoint['model'])
#     model.eval()

#     # Inference
#     with torch.no_grad():
#         for i, (fnames, imgs, masks, formulas) in enumerate(tqdm(test_loader, desc="Inference")):
#             if i >= 5:
#                 break
#             imgs, masks = imgs.to(device), masks.to(device)

#             hypotheses, batch_attention_weights, batch_feature_h, batch_feature_w = beam_search_batch(model, imgs, masks, beam_size=beam_size, max_len=max_len, alpha=1.0, vocab=vocab)
#             pred_seq = hypotheses[0]
#             sample_attention_weights = batch_attention_weights[0]
#             feature_h = batch_feature_h[0]
#             feature_w = batch_feature_w[0]

#             gt_seq = formulas[0]

#             pred_latex = vocab.indices2label(pred_seq)
#             with open(os.path.join(output_dir, f"{fnames[0]}.txt"), "w", encoding="utf-8") as f:
#                 f.write(f"%{fnames[0]}\n${pred_latex}$")

#             # Load original image
#             original_image_path = os.path.join(data_root, "img", f"{fnames[0]}.bmp")
#             original_image = Image.open(original_image_path).convert("RGB")

#             # Placeholder for best crop
#             best_crop = (0, 0, original_image.size[0], original_image.size[1])

#             # Process attention weights for visualization
#             processed_attention_weights = []
#             num_layers = len(sample_attention_weights[0])
#             num_heads = sample_attention_weights[0][0].shape[0]

#             for t in range(len(pred_latex.split())):
#                 token_attention_weights = [sample_attention_weights[t][layer_idx] for layer_idx in range(num_layers)]
#                 concatenated_attention = torch.cat(token_attention_weights, dim=0)  # Shape: (num_layers * num_heads, batch_size, 1, source_len)
#                 averaged_attention = torch.mean(concatenated_attention, dim=(0, 1))  # Average across layers/heads and batch
#                 averaged_attention = averaged_attention.squeeze()  # Shape: (source_len,)
#                 processed_attention_weights.append(averaged_attention)

#             # Visualize attention maps
#             visualize_attention_maps(original_image, processed_attention_weights, pred_latex.split(), best_crop, feature_h, feature_w)

def is_effectively_binary(img, threshold_percentage=0.9):
    dark_pixels = np.sum(img < 20)
    bright_pixels = np.sum(img > 235)
    total_pixels = img.size
    
    return (dark_pixels + bright_pixels) / total_pixels > threshold_percentage

def before_padding(image):
    
    # apply Canny edge detector to find text edges
    edges = cv2.Canny(image, 50, 150)

    # apply dilation to connect nearby edges
    kernel = np.ones((7, 13), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=8)

    # find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    
    # optimize crop rectangle using F1 score
    # sort components by number of white pixels (excluding background which is label 0)
    sorted_components = sorted(range(1, num_labels), 
                             key=lambda i: stats[i, cv2.CC_STAT_AREA], 
                             reverse=True)
    
    # Initialize with empty crop
    best_f1 = 0
    best_crop = (0, 0, image.shape[1], image.shape[0])
    total_white_pixels = np.sum(dilated > 0)

    current_mask = np.zeros_like(dilated)
    x_min, y_min = image.shape[1], image.shape[0]
    x_max, y_max = 0, 0
    
    for component_idx in sorted_components:
        # add this component to our mask
        component_mask = (labels == component_idx)
        current_mask = np.logical_or(current_mask, component_mask)
        
        # update bounding box
        comp_y, comp_x = np.where(component_mask)
        if len(comp_x) > 0 and len(comp_y) > 0:
            x_min = min(x_min, np.min(comp_x))
            y_min = min(y_min, np.min(comp_y))
            x_max = max(x_max, np.max(comp_x))
            y_max = max(y_max, np.max(comp_y))
        
        # calculate the current crop
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        crop_area = width * height
        

        crop_mask = np.zeros_like(dilated)
        crop_mask[y_min:y_max+1, x_min:x_max+1] = 1
        white_in_crop = np.sum(np.logical_and(dilated > 0, crop_mask > 0))
        
        # calculate F1 score
        precision = white_in_crop / crop_area
        recall = white_in_crop / total_white_pixels
        f1 = 2 * precision * recall / (precision + recall)
        
        if f1 > best_f1:
            best_f1 = f1
            best_crop = (x_min, y_min, x_max, y_max)
    
    # apply the best crop to the original image
    x_min, y_min, x_max, y_max = best_crop
    cropped_image = image[y_min:y_max+1, x_min:x_max+1]
    # cropped_image = cv2.add(cropped_image, 10)
    # cv2.imwrite('debug_process_img.jpg', cropped_image)

    
    # apply Gaussian adaptive thresholding
    if is_effectively_binary(cropped_image):
        _, thresh = cv2.threshold(cropped_image, 127, 255, cv2.THRESH_BINARY)
    else:
        thresh = cv2.adaptiveThreshold(
            cropped_image, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
    # cv2.imwrite('debug_process_img.jpg', thresh)
    
    # ensure background is black
    white = np.sum(thresh == 255)
    black = np.sum(thresh == 0)
    if white > black:
        thresh = 255 - thresh

    # add padding
    result = cv2.copyMakeBorder(
        thresh, 
        5, 
        5, 
        5, 
        5, 
        cv2.BORDER_CONSTANT, 
        value=0
    )
    
    return result, best_crop


inp_h = 256
inp_w = 256 * 8


def process_img(filename):
    """
    Load, binarize, ensures background is black, resize and apply centered padding
    """

    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    bin_img, best_crop = before_padding(image)

    h, w = bin_img.shape
    new_w = int((inp_h / h) * w)

    if new_w > inp_w:
        resized_img = cv2.resize(bin_img, (inp_w, inp_h), interpolation=cv2.INTER_AREA)
    else:
        resized_img = cv2.resize(bin_img, (new_w, inp_h), interpolation=cv2.INTER_AREA)
        padded_img = np.ones((inp_h, inp_w), dtype=np.uint8) * 0  # black background
        x_offset = (inp_w - new_w) // 2
        padded_img[:, x_offset:x_offset + new_w] = resized_img
        resized_img = padded_img

    # debugging only
    # resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
    # cv2.imwrite('debug_process_img.jpg', resized_img)
    return resized_img, best_crop


def main():
    # Config
    checkpoint_path = "checkpoints/bttr_best.pth"
    image_path = "real_img.png"
    caption_path = "resources/CROHME/test/caption.txt"
    output_dir = "./inference_results"
    beam_size = 10
    max_len = 200

    os.makedirs(output_dir, exist_ok=True)

    # Load vocab
    vocab = CROHMEVocab()

    # Load formula from caption.txt
    img_name = "UN_125_em_565"
    with open(caption_path, "r", encoding="utf-8") as f:
        lines = [line.strip().split() for line in f]
    formula_tokens = [tokens[1:] for tokens in lines if tokens[0] == img_name][0]

    # Load image
    from torchvision.transforms import ToTensor
    img = Image.open(image_path).convert("L")
    orig_image = Image.open(image_path).convert("RGB")
    tmp_img, best_crop = process_img(image_path)
    img_tensor = ToTensor()(tmp_img).unsqueeze(0).to(device)  # shape: [1, 1, H, W]
    mask = torch.zeros_like(img_tensor[:, 0], dtype=torch.bool).to(device)  # no padding

    # Load model
    model = BTTR(d_model=256, growth_rate=16, num_layers=3, nhead=8, num_decoder_layers=3, dim_feedforward=1024, dropout=0.3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Inference
    with torch.no_grad():
        hypotheses, batch_attention_weights, batch_feature_h, batch_feature_w = beam_search_batch(
            model, img_tensor, mask, beam_size=beam_size, max_len=max_len, alpha=1.0, vocab=vocab
        )
        pred_seq = hypotheses[0]
        sample_attention_weights = batch_attention_weights[0]
        feature_h = batch_feature_h[0]
        feature_w = batch_feature_w[0]

        pred_latex = vocab.indices2label(pred_seq)
        with open(os.path.join(output_dir, f"{img_name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"%{img_name}\n${pred_latex}$")

        # Prepare attention maps
        processed_attention_weights = []
        num_layers = len(sample_attention_weights[0])
        num_heads = sample_attention_weights[0][0].shape[0]

        for t in range(len(pred_latex.split())):
            token_attention_weights = [sample_attention_weights[t][layer_idx] for layer_idx in range(num_layers)]
            concatenated_attention = torch.cat(token_attention_weights, dim=0)
            averaged_attention = torch.mean(concatenated_attention, dim=(0, 1))
            averaged_attention = averaged_attention.squeeze()
            processed_attention_weights.append(averaged_attention)
            
        print(processed_attention_weights[0].shape)

        # Visualize
        visualize_attention_maps(orig_image, processed_attention_weights, pred_latex.split(),
                                 best_crop=best_crop)



def visualize_attention_maps(orig_image: Image, alphas, latex_tokens, best_crop, max_cols=4):
    '''
    Visualize attention maps over the original (unpadded) image
    '''
    orig_image = orig_image.crop(best_crop)
    orig_w, orig_h = orig_image.size
    ratio = inp_h / inp_w

    num_tokens = len(latex_tokens)
    num_cols = min(max_cols, num_tokens)
    num_rows = int(np.ceil(num_tokens / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, int(num_rows * 6 * orig_h / orig_w)))
    axes = np.array(axes).reshape(-1)

    for i, (token, alpha) in enumerate(zip(latex_tokens, alphas)):
        ax = axes[i]

        # alpha = alpha.squeeze(0)
        alpha_len = alpha.shape[0]
        alpha_w = int(np.sqrt(alpha_len / ratio))
        alpha_h = int(np.sqrt(alpha_len * ratio))

        # resize to (orig_h, interpolated_w)
        alpha = alpha.view(1, 1, alpha_h, alpha_w)
        interp_w = int(orig_h / ratio)

        alpha = F.interpolate(alpha, size=(orig_h, interp_w), mode='bilinear', align_corners=False)
        alpha = alpha.squeeze().cpu().numpy()

        # fix aspect ratio mismatch
        if interp_w > orig_w:
            # center crop width
            start = (interp_w - orig_w) // 2
            alpha = alpha[:, start:start + orig_w]
        elif interp_w < orig_w:
            # stretch to fit width
            alpha = cv2.resize(alpha, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        ax.imshow(orig_image)
        ax.imshow(alpha, cmap='jet', alpha=0.4)
        ax.set_title(f'{token}', fontsize=10 * 8 * orig_h / orig_w)
        ax.axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig('attention_maps_wap.png', bbox_inches='tight', dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
