import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import albumentations as A
from PIL import Image
import pandas as pd
import cv2
import numpy as np

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
    
    # clean up noise using median filter
    denoised = cv2.medianBlur(thresh, 3)
    for _ in range(7):
        denoised = cv2.medianBlur(denoised, 3)
    # cv2.imwrite('debug_process_img.jpg', denoised)

    # add padding
    result = cv2.copyMakeBorder(
        denoised, 
        5, 
        5, 
        5, 
        5, 
        cv2.BORDER_CONSTANT, 
        value=0
    )
    
    return result, best_crop


inp_h = 128
inp_w = 128 * 8


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
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
    # cv2.imwrite('debug_process_img.jpg', resized_img)
    return resized_img, best_crop


class HMERDataset(Dataset):
    '''
    Dataset for HMER
    '''

    def __init__(self, data_folder, label_file, vocab, transform=None, max_length=150):
        '''
        Initialize the dataset

        data_folder: folder containing images

        label_file: tsv file with two rows (filename, label), assuming no header

        vocab: Vocabulary object for tokenization

        transform: image transformations

        max_length: maximum sequence length
        '''
        self.data_folder = data_folder
        self.max_length = max_length
        self.vocab = vocab

        df = pd.read_csv(label_file, sep='\t', header=None, names=['filename', 'label'])
        df['filename'] = df['filename'].apply(lambda x: x + '.bmp')
        self.annotations = dict(zip(df['filename'], df['label']))

        # get image paths and captions
        self.image_paths = list(self.annotations.keys())

        # default transforms
        if transform is None:
            transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.pytorch.ToTensorV2()
            ])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # get image path and latex
        image_path = self.image_paths[idx]
        latex = self.annotations[image_path]

        # load and transform image
        # image = Image.open(os.path.join(self.data_folder, image_path)).convert('RGB')
        processed_img, _ = process_img(os.path.join(self.data_folder, image_path))
        image = np.array(Image.fromarray(processed_img).convert('RGB'))

        image = self.transform(image=image)['image']

        # tokenize latex
        tokens = self.vocab.tokenize(latex)

        # add start and end tokens
        tokens = [self.vocab.start_token] + tokens + [self.vocab.end_token]

        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        caption_length = torch.LongTensor([len(tokens)])
        # Pad to max length
        if len(tokens) < self.max_length:
            tokens = tokens + [self.vocab.pad_token] * (self.max_length - len(tokens))
            
        # Convert to tensor
        caption = torch.LongTensor(tokens)

        return image, caption, caption_length


class Vocabulary:
    '''
    Vocabulary class for tokenization
    '''

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        # add special tokens
        self.add_word('<pad>')  # padding token
        self.add_word('<start>')  # start token
        self.add_word('<end>')  # end token
        self.add_word('<unk>')  # unknown token

        self.pad_token = self.word2idx['<pad>']
        self.start_token = self.word2idx['<start>']
        self.end_token = self.word2idx['<end>']
        self.unk_token = self.word2idx['<unk>']

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)

    def tokenize(self, latex):
        '''
        Tokenize latex string into indices. This assumes the tokens are separated by space.
        '''
        tokens = []

        for char in latex.split():
            if char in self.word2idx:
                tokens.append(self.word2idx[char])
            else:
                tokens.append(self.unk_token)

        return tokens

    def build_vocab(self, label_file):
        '''Build vocabulary from label file'''
        df = pd.read_csv(label_file, sep='\t', header=None,
                         names=['filename', 'label'])
        all_labels_text = ' '.join(df['label'].astype(str).tolist())
        tokens = sorted(set(all_labels_text.split()))
        for char in tokens:
            self.add_word(char)


# def main() -> None:
#     # test vocab
#     vocab = Vocabulary()
#     vocab.build_vocab('resources/CROHME/train/caption.txt')
#     print(vocab.idx2word.items())


def main() -> None:
    # for testing dataset only, do not copy

    batch_size = 32

    vocab = Vocabulary()
    for split in ['2014', '2016', '2019', 'train']:
        vocab.build_vocab(f'resources/CROHME/{split}/caption.txt')

    train_dataset_1 = HMERDataset(
        data_folder='resources/CROHME/train/img',
        label_file='resources/CROHME/train/caption.txt',
        vocab=vocab
    )

    train_dataset_2 = HMERDataset(
        data_folder='resources/CROHME/2014/img',
        label_file='resources/CROHME/2014/caption.txt',
        vocab=vocab
    )

    train_dataset = ConcatDataset([train_dataset_1, train_dataset_2])
    val_dataset = HMERDataset(
        data_folder='resources/CROHME/2016/img',
        label_file='resources/CROHME/2016/caption.txt',
        vocab=vocab
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


if __name__ == '__main__':
    main()
