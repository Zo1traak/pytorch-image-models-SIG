"""Multi-label dataset for ILS interference spectrogram classification.

Reads CSV label files and loads corresponding spectrogram images.
"""
import csv
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

LABEL_NAMES = ['Arc', 'HV', 'IR', 'Jam', 'Lamp']


def get_transforms(img_size: int = 224, is_train: bool = True):
    """Return torchvision transforms for train or val/test."""
    import torchvision.transforms as T
    if is_train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize(int(img_size * 1.143)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class ILSInterferenceDataset(Dataset):
    """Multi-label dataset reading spectrogram images and CSV labels.

    Expected directory structure:
        split_name/
            images/       # spectrogram PNG files
            labels.csv    # columns: filename, ..., Arc, HV, IR, Jam, Lamp, ...
    """

    def __init__(self, root: str, split: str = 'train', transform=None):
        self.root = Path(root) / split
        self.transform = transform

        csv_path = self.root / 'labels.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f'Labels file not found: {csv_path}')

        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            self.rows = list(reader)

        self.img_dir = self.root / 'images'

        # Build list of (filename, label_tensor) tuples
        self.samples = []
        for row in self.rows:
            filename = row['filename']
            labels = torch.tensor(
                [int(row[name]) for name in LABEL_NAMES],
                dtype=torch.float32,
            )
            self.samples.append((filename, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, labels = self.samples[idx]
        img_path = self.img_dir / filename
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, labels

    @property
    def num_classes(self):
        return len(LABEL_NAMES)

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([x[0] for x in batch])
        labels = torch.stack([x[1] for x in batch])
        return images, labels
