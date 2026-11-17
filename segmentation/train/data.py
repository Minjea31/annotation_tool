import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset

import _bootstrap  # noqa: F401
from autorace_lane_lite.model import CLASSES, MEAN, STD


class LaneDataset(Dataset):
    def __init__(self, root, split, width, height, augment=False):
        self.root = Path(root)
        manifest = json.loads((self.root / 'manifest.json').read_text())
        if tuple(manifest['classes']) != CLASSES or manifest['ignore_index'] != 255:
            raise ValueError('Dataset class mapping mismatch')
        self.rows = [row for row in manifest['samples'] if row['split'] == split]
        if not self.rows:
            raise ValueError(f'Missing/empty {split} split; use separate capture sessions for validation')
        self.size, self.augment = (width, height), augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['image']) as source:
            image = source.convert('RGB')
        with Image.open(self.root / row['mask']) as source:
            mask = source.copy()
        if self.augment:
            if torch.rand(()).item() < 0.5:
                image, mask = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = ImageEnhance.Brightness(image).enhance(0.7 + 0.6 * torch.rand(()).item())
            image = ImageEnhance.Contrast(image).enhance(0.8 + 0.4 * torch.rand(()).item())
        rgb = cv2.resize(np.asarray(image), self.size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        rgb = (rgb - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
        target = cv2.resize(np.array(mask), self.size, interpolation=cv2.INTER_NEAREST).astype(np.int64)
        if not np.isin(target, (0, 1, 2, 255)).all():
            raise ValueError('Mask must contain class indices 0/1/2/255, not colored pixels')
        return torch.from_numpy(rgb.transpose(2, 0, 1).copy()), torch.from_numpy(target)
