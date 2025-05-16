import os
import torch
import pandas as pd
from PIL import Image
import cv2
import albumentations as A
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json

try:
    from .wap_dataloader import Vocabulary, process_img, inp_h, inp_w
    torch.serialization.add_safe_globals([Vocabulary])
except:
    pass

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# import model
try:
    from .wap import WAP
except:
    pass

embed_size = 256
encoder_dim = 256
decoder_dim = 512
attention_dim = 256
dropout = 0.5
grad_clip = 5.0
lbd = 0.5

def levenshtein_distance(lst1, lst2):
    m = len(lst1)
    n = len(lst2)

    prev_row = [j for j in range(n + 1)]
    curr_row = [0] * (n + 1)
    for i in range(1, m + 1):
        curr_row[0] = i

        for j in range(1, n + 1):
            if lst1[i - 1] == lst2[j - 1]:
                curr_row[j] = prev_row[j - 1]
            else:
                curr_row[j] = 1 + min(
                    curr_row[j - 1],
                    prev_row[j],
                    prev_row[j - 1]
                )

        prev_row = curr_row.copy()
    return curr_row[n]

def load_checkpoint(checkpoint_path, device):
    '''
    Load checkpoint and return model and vocabulary
    '''
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    vocab = checkpoint['vocab']
    
    model = WAP(
        vocab_size=len(vocab),
        embed_size=embed_size,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        attention_dim=attention_dim,
        dropout=dropout
    ).to(device)
    
    model.load_state_dict(checkpoint['model'])
    
    return model, vocab

def recognize_single_image(model, image_path, vocab, device, max_length=150, visualize_attention=False):
    '''
    Recognize handwritten mathematical expression from a single image
    '''
    # prepare image
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.pytorch.ToTensorV2()
    ])
    
    # load and transform image
    image = Image.open(image_path).convert('RGB')
    tmp_img, best_crop = process_img(image_path)
    image_processed = np.array(Image.fromarray(tmp_img).convert('RGB'))
    image_tensor = transform(image=image_processed)['image'].unsqueeze(0).to(device)
    

    model.eval()
    with torch.no_grad():
        # generate LaTeX
        predictions, alphas = model.recognize(
            image_tensor,
            max_length=max_length,
            start_token=vocab.start_token,
            end_token=vocab.end_token
        )
    
    # convert indices to LaTeX tokens
    latex_tokens = []
    for idx in predictions:
        if idx == vocab.end_token:
            break
        latex_tokens.append(vocab.idx2word[idx])
    
    # join tokens to get complete LaTeX
    latex = ' '.join(latex_tokens)
    
    # visualize attention
    if visualize_attention:
        visualize_attention_maps(image, alphas, latex_tokens, best_crop)
    
    return latex


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

        alpha = alpha.squeeze(0)
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


def evaluate_model(model, test_folder, label_file, vocab, device, max_length=150):
    '''
    Evaluate model on test set
    '''

    df = pd.read_csv(label_file, sep='\t', header=None, names=['filename', 'label'])
    df['filename'] = df['filename'].apply(lambda x: x + '.bmp')
    annotations = dict(zip(df['filename'], df['label']))
    
    model.eval()
    
    correct = 0
    err1 = 0
    err2 = 0
    err3 = 0
    total = 0

    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.pytorch.ToTensorV2()
    ])
    
    results = {}
    
    for image_path, gt_latex in tqdm(annotations.items()):
        gt_latex: str = gt_latex
        # image = Image.open(os.path.join(test_folder, image_path)).convert('RGB')
        tmp_img, _ = process_img(os.path.join(test_folder, image_path))
        processed_img = np.array(Image.fromarray(tmp_img).convert('RGB'))
        image_tensor = transform(image=processed_img)['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions, _ = model.recognize(
                image_tensor,
                max_length=max_length,
                start_token=vocab.start_token,
                end_token=vocab.end_token
            )
        
        # convert indices to LaTeX tokens
        pred_latex_tokens = []
        for idx in predictions:
            if idx == vocab.end_token:
                break
            pred_latex_tokens.append(vocab.idx2word[idx])
        
        pred_latex = ' '.join(pred_latex_tokens)

        gt_latex_tokens = gt_latex.split()
        edit_distance = levenshtein_distance(pred_latex_tokens, gt_latex_tokens)
        
        if edit_distance == 0:
            correct += 1
        if edit_distance == 1:
            err1 += 1
        if edit_distance == 2:
            err2 += 1
        if edit_distance == 3:
            err3 += 1
        
        total += 1
        
        # save result
        results[image_path] = {
            'ground_truth': gt_latex,
            'prediction': pred_latex,
            'edit_distance': edit_distance
        }
    
    # Calculate accuracy
    exprate = round(correct / total, 4)
    exprate_leq1 = round((correct + err1) / total, 4)
    exprate_leq2 = round((correct + err1 + err2) / total, 4)
    exprate_leq3 = round((correct + err1 + err2 + err3) / total, 4)
    
    # Save results to file
    with open('evaluation_results_wap.json', 'w', encoding='utf-8') as f:
        json.dump({
            'accuracy': [exprate, exprate_leq1, exprate_leq2, exprate_leq3],
            'results': results
        }, f, indent=4)
    
    return [exprate, exprate_leq1, exprate_leq2, exprate_leq3], results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    checkpoint_path = 'checkpoints/wap_best_0.49.pth'
    mode = 'single' # 'single' or 'evaluate'

    # for single mode

    image_path = 'resources/CROHMEv2/test/img/UN_125_em_565.bmp'
    # image_path = 'temp_input_image.png'
    visualize = True

    # for evaluation mode
    test_folder = 'resources/CROHMEv2/test/img'
    label_file = 'resources/CROHMEv2/test/caption.txt'

    # load model and vocabulary
    model, vocab = load_checkpoint(checkpoint_path, device)
    
    if mode == 'single':
        if image_path is None:
            raise ValueError('Image path is required for single mode')
        
        latex = recognize_single_image(model, image_path, vocab, device, visualize_attention=visualize)
        
        print(f'Recognized LaTeX: {latex}')
        
    elif mode == 'evaluate':
        if test_folder is None or label_file is None:
            raise ValueError('Test folder and annotation file are required for evaluate mode')

        exprate, results = evaluate_model(model, test_folder, label_file, vocab, device)
        
        print(f'ExpRate: {exprate}')

if __name__ == '__main__':
    from wap_dataloader import Vocabulary, process_img, inp_h, inp_w
    torch.serialization.add_safe_globals([Vocabulary])
    from wap import WAP
    main()