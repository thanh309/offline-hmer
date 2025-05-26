import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence
import time
import wandb
from datetime import datetime
import albumentations as A
import cv2
import random
from PIL import Image
import pandas as pd
import torchvision.models as models
import torch.nn.functional as F

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
my_secret = user_secrets.get_secret("wandb_api_key") 
wandb.login(key=my_secret)

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
    for _ in range(3):
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


class Vocabulary:
    '''
    Vocabulary class for tokenization
    '''
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
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
        self.image_paths = list(self.annotations.keys())

        if transform is None:
            transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.pytorch.ToTensorV2()
            ])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        latex = self.annotations[image_path]
        processed_img, _ = process_img(os.path.join(self.data_folder, image_path))
        image = np.array(Image.fromarray(processed_img).convert('RGB'))
        image = self.transform(image=image)['image']
        tokens = self.vocab.tokenize(latex)
        tokens = [self.vocab.start_token] + tokens + [self.vocab.end_token]

        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        caption_length = torch.LongTensor([len(tokens)])
        if len(tokens) < self.max_length:
            tokens = tokens + [self.vocab.pad_token] * (self.max_length - len(tokens))

        caption = torch.LongTensor(tokens)

        return image, caption, caption_length


class EncoderCNN(nn.Module):
    """
    CNN-based encoder for extracting visual features from handwritten math expressions
    """
    def __init__(self, enc_hidden_size=256):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet18(weights=None)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.conv_reduce = nn.Conv2d(512, enc_hidden_size, kernel_size=1)

    def forward(self, images):
        """
        Extract features from input images; then reshape for attention mechanism
        images: [batch_size, channels, height, width]
        """
        features = self.resnet(images)
        features = self.conv_reduce(features)
        batch_size = features.size(0)
        feature_size = features.size(1)
        height, width = features.size(2), features.size(3)
        features = features.permute(0, 2, 3, 1).contiguous()
        features = features.view(batch_size, height*width, feature_size)
        return features


class BahdanauAttention(nn.Module):
    """
    Bahdanau attention mechanism with coverage
    encoder_att: projects encoder output to the attention dimension
    decoder_att: projects decoder hidden state to the attention dimension
    coverage_att: projects the coverage vector to the attention dimension
    full_att: computes a scalar attention score from the combined feature
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.coverage_att = nn.Linear(1, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden, coverage=None):
        """
        Calculate context vector for the current time step
        encoder_out: [batch_size, num_pixels, encoder_dim]
        decoder_hidden: [batch_size, decoder_dim]
        coverage: [batch_size, num_pixels, 1]
        """
        num_pixels = encoder_out.size(1)
        encoder_att = self.encoder_att(encoder_out)
        decoder_att = self.decoder_att(decoder_hidden)
        decoder_att = decoder_att.unsqueeze(1)

        if coverage is None:
            coverage = torch.zeros(encoder_out.size(0), num_pixels, 1).to(encoder_out.device)
        coverage_att = self.coverage_att(coverage)

        att = torch.tanh(encoder_att + decoder_att + coverage_att)
        att = self.full_att(att).squeeze(2)
        alpha = F.softmax(att, dim=1)
        context_vector = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        coverage = coverage + alpha.unsqueeze(2)

        return context_vector, alpha, coverage


class DecoderRNN(nn.Module):
    """
    LSTM-based decoder with attention
    """
    def __init__(self, vocab_size, embed_size, encoder_dim, decoder_dim, attention_dim, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = BahdanauAttention(
            encoder_dim, decoder_dim, attention_dim)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden_state(self, encoder_out):
        """
        Initialize hidden state and cell state for the LSTM
        encoder_out: [batch_size, num_pixels, encoder_dim]
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward pass for training
        encoder_out: [batch_size, num_pixels, encoder_dim]
        encoded_captions: [batch_size, max_caption_length]
        caption_lengths: [batch_size, 1]
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        embeddings = self.embedding(encoded_captions)
        h, c = self.init_hidden_state(encoder_out)
        decode_lengths = (caption_lengths - 1).tolist()
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)
        coverage = torch.zeros(batch_size, num_pixels, 1).to(encoder_out.device)
        coverage_seq = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            context_vector, alpha, coverage = self.attention(
                encoder_out[:batch_size_t],
                h[:batch_size_t],
                coverage[:batch_size_t] if t > 0 else None
            )
            coverage_seq[:batch_size_t, t, :] = coverage.squeeze(2)
            gate = torch.sigmoid(self.f_beta(h[:batch_size_t]))
            context_vector = gate * context_vector
            h, c = self.lstm_cell(
                torch.cat([embeddings[:batch_size_t, t, :], context_vector], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, alphas, coverage_seq, decode_lengths, sort_ind

    def generate_caption(self, encoder_out, max_length=150, start_token=1, end_token=2):
        """
        Generate captions (LaTeX sequences)
        note: prediction will not have start_token
        encoder_out: [1, num_pixels, encoder_dim]
        """
        batch_size = encoder_out.size(0)
        assert batch_size == 1, "batch prediction is not supported"
        predictions = []
        alphas = []
        h, c = self.init_hidden_state(encoder_out)
        prev_word = torch.LongTensor([start_token]).to(encoder_out.device)
        coverage = None

        for i in range(max_length):
            embeddings = self.embedding(prev_word)
            context_vector, alpha, coverage = self.attention(
                encoder_out,
                h,
                coverage
            )
            gate = torch.sigmoid(self.f_beta(h))
            context_vector = gate * context_vector
            h, c = self.lstm_cell(
                torch.cat([embeddings, context_vector], dim=1),
                (h, c)
            )
            preds = self.fc(h)
            _, next_word = torch.max(preds, dim=1)
            predictions.append(next_word.item())
            alphas.append(alpha)
            prev_word = next_word
            if next_word.item() == end_token:
                break

        return predictions, alphas


class WAP(nn.Module):
    """
    Watch, Attend and Parse model for handwritten mathematical expression recognition
    """
    def __init__(self, vocab_size, embed_size=256, encoder_dim=256, decoder_dim=512, attention_dim=256, dropout=0.5):
        super(WAP, self).__init__()
        self.encoder = EncoderCNN(enc_hidden_size=encoder_dim)
        self.decoder = DecoderRNN(
            vocab_size=vocab_size,
            embed_size=embed_size,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim,
            dropout=dropout
        )

    def forward(self, images, encoded_captions, caption_lengths):
        """
        Forward pass
        images: [batch_size, channels, height, width]
        encoded_captions: [batch_size, max_caption_length]
        caption_lengths: [batch_size, 1]
        """
        encoder_out = self.encoder(images)
        predictions, alphas, coverage_seq, decode_lengths, sort_ind = self.decoder(
            encoder_out, encoded_captions, caption_lengths
        )
        return predictions, alphas, coverage_seq, decode_lengths, sort_ind

    def recognize(self, image, max_length=150, start_token=1, end_token=2):
        """
        Recognize handwritten mathematical expression and output LaTeX sequence
        image: [1, channels, height, width]
        """
        batch_size = image.size(0)
        assert batch_size == 1, "batch prediction is not supported"
        encoder_out = self.encoder(image)
        predictions, alphas = self.decoder.generate_caption(
            encoder_out,
            max_length=max_length,
            start_token=start_token,
            end_token=end_token
        )
        return predictions, alphas


class RandomMorphology(A.ImageOnlyTransform):
    def __init__(self, p=0.5, kernel_size=3):
        super(RandomMorphology, self).__init__(p)
        self.kernel_size = kernel_size

    def apply(self, img, **params):
        op = random.choice(['erode', 'dilate'])
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        if op == 'erode':
            return cv2.erode(img, kernel, iterations=1)
        else:
            return cv2.dilate(img, kernel, iterations=1)

train_transforms = A.Compose([
    A.Rotate(limit=5, p=0.25, border_mode=cv2.BORDER_REPLICATE),
    A.ElasticTransform(alpha=100, sigma=7, p=0.5, interpolation=cv2.INTER_CUBIC),
    RandomMorphology(p=0.5, kernel_size=2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.pytorch.ToTensorV2()
])


def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=5.0, lbd=0.5, print_freq=10):
    '''
    Train the model for one epoch
    '''
    model.train()
    losses = []

    for i, (images, captions, caption_lengths) in enumerate(train_loader):
        images = images.to(device)
        captions = captions.to(device)
        caption_lengths = caption_lengths.to(device)

        predictions, alphas, coverage_seq, decode_lengths, sort_ind = model(
            images, captions, caption_lengths
        )

        targets = captions[sort_ind, 1:]

        predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        loss = criterion(predictions, targets)

        coverage_loss = torch.mean(torch.sum(torch.min(alphas, coverage_seq), dim=1))
        loss += lbd * coverage_loss

        optimizer.zero_grad()
        loss.backward()

        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        losses.append(loss.item())

    return sum(losses) / len(losses)


def validate(model, val_loader, criterion, device, lbd=0.5):
    '''
    Validate the model
    '''
    model.eval()
    losses = []

    with torch.no_grad():
        for i, (images, captions, caption_lengths) in enumerate(val_loader):
            images = images.to(device)
            captions = captions.to(device)
            caption_lengths = caption_lengths.to(device)

            predictions, alphas, coverage_seq, decode_lengths, sort_ind = model(
                images, captions, caption_lengths
            )

            targets = captions[sort_ind, 1:]

            predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            loss = criterion(predictions, targets)

            coverage_loss = torch.mean(
                torch.sum(torch.min(alphas, coverage_seq), dim=1))
            loss += lbd * coverage_loss

            losses.append(loss.item())

    return sum(losses) / len(losses)


def main():

    dataset_dir = '/kaggle/input/dataset-name'
    splits = ['train', 'val', 'test']

    seed = 1337
    checkpoints_dir = '/kaggle/working/checkpoints'
    batch_size = 32

    embed_size = 256
    encoder_dim = 256
    decoder_dim = 512
    attention_dim = 256
    dropout = 0.5
    grad_clip = 5.0
    lbd = 0.5

    lr = 1e-4
    epochs = 100
    data_fractions = 1
    assert 0 < data_fractions <= 1, 'invalid data fractions'

    T_0 = 5
    T_mult = 2

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    os.makedirs(checkpoints_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')


    vocab = Vocabulary()
    for split in splits:
         vocab.build_vocab(f'{dataset_dir}/{split}/caption.txt')


    train_dataset = HMERDataset(
        data_folder=f'{dataset_dir}/train/img',
        label_file=f'{dataset_dir}/train/caption.txt',
        vocab=vocab,
        transform=train_transforms
    )

    val_dataset = HMERDataset(
        data_folder=f'{dataset_dir}/val/img',
        label_file=f'{dataset_dir}/val/caption.txt',
        vocab=vocab
    )

    sample_train = torch.randperm(len(train_dataset))[:int(len(train_dataset)*data_fractions)]
    sample_val = torch.randperm(len(val_dataset))[:int(len(val_dataset)*data_fractions)]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        sampler=SubsetRandomSampler(sample_train),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        sampler=SubsetRandomSampler(sample_val),
        drop_last=True
    )


    model = WAP(
        vocab_size=len(vocab),
        embed_size=embed_size,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        attention_dim=attention_dim,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_mult
    )

    run_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    wandb.init(project='offline-hmer', name=run_name, config={
        'seed': seed,
        'batch_size': batch_size,
        'embed_size': embed_size,
        'encoder_dim': encoder_dim,
        'decoder_dim': decoder_dim,
        'attention_dim': attention_dim,
        'dropout': dropout,
        'grad_clip': grad_clip,
        'lbd': lbd,
        'lr': lr,
        'epochs': epochs,
        'data_fractions': data_fractions,
        'T_0': T_0,
        'T_mult': T_mult
    })

    best_val_loss = float('inf')

    for epoch in range(epochs):
        curr_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1:03}/{epochs:03}')
        t1 = time.time()

        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=grad_clip,
            lbd=lbd,
            print_freq=1e6
        )

        val_loss = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            lbd=lbd
        )

        scheduler.step()
        t2 = time.time()

        print(f'train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, time: {t2 - t1:.4f} seconds')

        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate':curr_lr,
            'epoch': epoch
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
                'vocab': vocab
            }
            torch.save(checkpoint, os.path.join(checkpoints_dir, 'wap_best.pth'))
            print('model saved!')

    print('training completed!')
    wandb.finish()


main()
