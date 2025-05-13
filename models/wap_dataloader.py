import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import pandas as pd


class HMERDataset(Dataset):
    """
    Dataset for HMER
    """

    def __init__(self, data_folder, label_file, vocab, transform=None, max_length=150):
        """
        Initialize the dataset

        data_folder: folder containing images

        label_file: tsv file with two rows (filename, label), assuming no header

        vocab: Vocabulary object for tokenization

        transform: image transformations

        max_length: maximum sequence length
        """
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
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # get image path and latex
        image_path = self.image_paths[idx]
        latex = self.annotations[image_path]

        # load and transform image
        image = Image.open(os.path.join(self.data_folder, image_path)).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # tokenize latex
        tokens = self.vocab.tokenize(latex)

        # add start and end tokens
        tokens = [self.vocab.start_token] + tokens + [self.vocab.end_token]

        # # Pad to max length
        # if len(tokens) < self.max_length:
        #     tokens = tokens + [self.vocab.pad_token] * \
        #         (self.max_length - len(tokens))
        # else:
        #     tokens = tokens[:self.max_length]

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
    """
    Vocabulary class for tokenization
    """

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
        """
        Tokenize latex string into indices. This assumes the tokens are separated by space.
        """
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
    # test dataset

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
