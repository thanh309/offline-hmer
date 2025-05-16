import shutil
import random
from pathlib import Path


def merge_and_split_crohme(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    random.seed(seed)

    data = []
    input_dir = Path(input_dir)
    years = ['2014', '2016', '2019', 'train']

    for year in years:
        caption_file = input_dir / year / 'caption.txt'
        img_dir = input_dir / year / 'img'

        with open(caption_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name, latex = parts
                img_path = img_dir / f"{img_name}.bmp"
                data.append((img_name, latex, img_path))

    img_names = [entry[0] for entry in data]
    assert len(img_names) == len(set(img_names)), 'duplicate image found'

    random.shuffle(data)
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True)
        (split_dir / 'img').mkdir()

    for split, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        split_img_dir = output_dir / split / 'img'
        caption_path = output_dir / split / 'caption.txt'

        with open(caption_path, 'w', encoding='utf-8') as f:
            for img_name, latex, img_path in split_data:
                shutil.copy(img_path, split_img_dir / f"{img_name}.bmp")
                f.write(f"{img_name}\t{latex}\n")

    print(f"Train: {len(train_data)} samples")
    print(f"Validation: {len(val_data)} samples")
    print(f"Test: {len(test_data)} samples")


if __name__ == "__main__":
    input_dir = "resources/CROHME"
    output_dir = "resources/CROHMEv2"
    merge_and_split_crohme(input_dir, output_dir)
