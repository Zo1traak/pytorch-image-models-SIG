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


class ArcAwareSpecAugment:
    """Time-frequency masking that protects Arc transient bursts.

    Reduces time masking probability for samples containing Arc (short
    impulsive events that would be destroyed by aggressive time masking).
    Frequency masking is preserved for robustness against narrowband
    interference.

    Modes:
        light:   freq=8, time=15, num=1, Arc time prob 0.1
        current: freq=16, time=30, num=2, Arc time prob 0.5 (default)
        strong:  freq=24, time=45, num=3, Arc time prob 1.0
    """

    _PRESETS = {
        'light':   dict(freq_mask=8,  time_mask=15, num_freq=1, num_time=1,
                        arc_time_prob=0.1),
        'current': dict(freq_mask=16, time_mask=30, num_freq=2, num_time=2,
                        arc_time_prob=0.5),
        'strong':  dict(freq_mask=24, time_mask=45, num_freq=3, num_time=3,
                        arc_time_prob=1.0),
    }

    def __init__(self, mode: str = 'current'):
        cfg = self._PRESETS[mode]
        self.freq_mask_param = cfg['freq_mask']
        self.time_mask_param = cfg['time_mask']
        self.num_freq_masks = cfg['num_freq']
        self.num_time_masks = cfg['num_time']
        self.arc_time_prob = cfg['arc_time_prob']

    def __call__(self, img: torch.Tensor, has_arc: bool = False) -> torch.Tensor:
        """Apply time/frequency masking.

        Args:
            img: Normalized tensor [C, H, W]
            has_arc: If True, reduce time masking probability to protect
                     Arc's short-burst transient structure.
        """
        c, h, w = img.shape
        # Frequency masking (always applied, helps narrowband robustness)
        for _ in range(self.num_freq_masks):
            f_width = torch.randint(1, self.freq_mask_param + 1, (1,)).item()
            f_start = torch.randint(0, max(1, h - f_width), (1,)).item()
            img[:, f_start:f_start + f_width, :] = 0.0
        # Time masking with Arc-aware probability
        for _ in range(self.num_time_masks):
            if has_arc and torch.rand(1).item() > self.arc_time_prob:
                continue  # Skip time mask to protect Arc burst
            t_width = torch.randint(1, self.time_mask_param + 1, (1,)).item()
            t_start = torch.randint(0, max(1, w - t_width), (1,)).item()
            img[:, :, t_start:t_start + t_width] = 0.0
        return img


def get_transforms(img_size: int = 224, is_train: bool = True):
    """Return torchvision transforms for train or val/test.

    Note: SpecAugment is applied separately in the dataset, not via transforms.
    """
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

    def __init__(self, root: str, split: str = 'train', transform=None,
                 specaug: ArcAwareSpecAugment = None):
        self.root = Path(root) / split
        self.transform = transform
        self.specaug = specaug

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

        # Apply Arc-aware SpecAugment after normalization
        if self.specaug is not None:
            has_arc = labels[0].item() == 1.0  # Arc is index 0
            img = self.specaug(img, has_arc=has_arc)

        return img, labels

    @property
    def num_classes(self):
        return len(LABEL_NAMES)

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([x[0] for x in batch])
        labels = torch.stack([x[1] for x in batch])
        return images, labels
