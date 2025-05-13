import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence

# Import model and data loader from previous files
from wap import WAP
from wap_dataloader import HMERDataset, Vocabulary


import albumentations as A
import cv2
import random

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
    A.Rotate(limit=3, p=0.5, border_mode=cv2.BORDER_REPLICATE),
    A.ElasticTransform(alpha=50, sigma=5, p=0.3),
    A.Perspective(scale=(0.03, 0.05), p=0.3, keep_size=True),
    RandomMorphology(p=0.3, kernel_size=1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.pytorch.ToTensorV2()
])

def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=5.0, lbd=0.5, print_freq=10):
    """
    Train the model for one epoch
    """
    model.train()
    losses = []

    for i, (images, captions, caption_lengths) in enumerate(train_loader):
        images = images.to(device)
        captions = captions.to(device)
        caption_lengths = caption_lengths.to(device)

        # forward pass
        predictions, alphas, coverage_seq, decode_lengths, sort_ind = model(
            images, captions, caption_lengths
        )

        # calculate loss
        targets = captions[sort_ind, 1:]  # remove start token from targets

        # pack predictions for loss calculation
        predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        loss = criterion(predictions, targets)

        # add coverage loss
        # sum over time steps, mean over batch
        coverage_loss = torch.mean(torch.sum(torch.min(alphas, coverage_seq), dim=1))
        loss += lbd * coverage_loss

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # clip gradients
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # update weights
        optimizer.step()

        # track loss
        losses.append(loss.item())

        # print progress
        if i % print_freq == 0:
            print(f'Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}')

    return sum(losses) / len(losses)

def validate(model, val_loader, criterion, device, lbd=0.5):
    """
    Validate the model
    """
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
    splits = ['2014', '2016', '2019', 'train']
    dataset_dir = 'resources/CROHME'

    seed = 42
    checkpoints_dir = 'checkpoints'
    batch_size = 32

    embed_size = 256
    encoder_dim = 256
    decoder_dim = 512
    attention_dim = 256
    dropout = 0.5
    grad_clip = 5.0
    lbd = 0.5

    lr = 5e-4
    epochs = 50


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

    train_dataset_1 = HMERDataset(
        data_folder=f'{dataset_dir}/train/img',
        label_file=f'{dataset_dir}/train/caption.txt',
        vocab=vocab,
        transform=train_transforms
    )

    train_dataset_2 = HMERDataset(
        data_folder=f'{dataset_dir}/2014/img',
        label_file=f'{dataset_dir}/2014/caption.txt',
        vocab=vocab,
        transform=train_transforms
    )

    train_dataset = ConcatDataset([train_dataset_1, train_dataset_2])
    val_dataset = HMERDataset(
        data_folder=f'{dataset_dir}/2016/img',
        label_file=f'{dataset_dir}/2016/caption.txt',
        vocab=vocab
    )

    fractions = 0.05
    assert 0 < fractions <= 1, 'invalid fractions'
    sample_train = torch.randperm(len(train_dataset))[:int(len(train_dataset)*fractions)]
    sample_val = torch.randperm(len(val_dataset))[:int(len(val_dataset)*fractions)]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=False,
        sampler=SubsetRandomSampler(sample_train)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=False,
        sampler=SubsetRandomSampler(sample_val)
    )

    # create model
    model = WAP(
        vocab_size=len(vocab),
        embed_size=embed_size,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        attention_dim=attention_dim,
        dropout=dropout
    ).to(device)

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )

    # training loop
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f'Epoch {epoch+1:03}/{epochs:03}')

        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=grad_clip,
            lbd=lbd
        )

        val_loss = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            lbd=lbd
        )

        # update learning rate
        scheduler.step(val_loss)

        print(f'train loss: {train_loss:.4f}, val loss: {val_loss:.4f}')

        # save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
                'vocab': vocab
            }
            # torch.save(checkpoint, os.path.join(checkpoints_dir, 'wap_best.pth'))
            # print('model saved!')

    print('training completed!')


if __name__ == '__main__':
    main()
