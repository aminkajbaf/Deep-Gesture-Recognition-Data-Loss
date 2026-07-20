"""Dataset loader for DRAI sequences with simulated data-loss (random zero-out)."""

from __future__ import annotations

import os
import random
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

GESTURE_CLASSES = {
    "Swipe to Right": 0,
    "Swipe to Left": 1,
    "Pull": 2,
    "Chek": 3,
    "Double Push": 4,
    "Rotate CCW": 5,
    "Rotate CW": 6,
    "Moving Finger": 7,
    "Double Hand Push": 8,
    "Cross": 9,
    "Swipe to Forward and Backward": 10,
    "Push": 11,
    "Motion": 12,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _parse_label(path: str) -> int:
    basename = os.path.basename(path).replace(".npy", "")
    if "Motion" in basename or basename.startswith("n_"):
        return GESTURE_CLASSES["Motion"]
    parts = basename.split("_")
    # Prefer token that matches a known class name
    for token in parts[1:]:
        if token in GESTURE_CLASSES:
            return GESTURE_CLASSES[token]
    # Common naming: prefix_id_Gesture Name.npy (gesture may contain spaces)
    if len(parts) >= 3:
        gesture = "_".join(parts[2:]) if "_" in basename else parts[2]
        # Recover original spaces from filenames that used spaces
        gesture = os.path.basename(path).replace(".npy", "").split("_", 2)[-1]
        if gesture in GESTURE_CLASSES:
            return GESTURE_CLASSES[gesture]
    raise KeyError(f"Cannot parse gesture class from filename: {path}")


class GestureDataset(Dataset):
    """
    Loads radar DRAI tensors from .npy files.

    Returns:
        clean   : normalized tensor (C, H, W)
        corrupt : clean with random zero-out (simulated DL)
        label   : gesture class id
    """

    def __init__(
        self,
        root_dirs: Sequence[str],
        negative_root: Optional[str] = None,
        transform: Optional[Callable] = None,
        dl_ratio: float = 0.2,
        target_frames: int = 20,
    ):
        self.transform = transform
        self.dl_ratio = dl_ratio
        self.target_frames = target_frames
        self.classes = GESTURE_CLASSES

        self.file_paths: List[str] = []
        for root in root_dirs:
            self.file_paths.extend(
                os.path.join(root, f) for f in os.listdir(root) if f.endswith(".npy")
            )
        if negative_root:
            self.file_paths.extend(
                os.path.join(negative_root, f)
                for f in os.listdir(negative_root)
                if f.endswith(".npy")
            )

        self.labels = [_parse_label(p) for p in self.file_paths]

    def __len__(self) -> int:
        return len(self.file_paths)

    def _resize_time(self, data: torch.Tensor) -> torch.Tensor:
        t = data.shape[0]
        if t == self.target_frames:
            return data
        if t > self.target_frames:
            idx = torch.linspace(0, t - 1, self.target_frames).long()
            return data[idx]
        pad = torch.zeros(self.target_frames - t, *data.shape[1:], dtype=data.dtype)
        return torch.cat([data, pad], dim=0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        data = torch.from_numpy(np.load(self.file_paths[idx]).astype(np.float32))
        data = self._resize_time(data)

        # Min-max normalize per sample
        dmin, dmax = data.min(), data.max()
        data = (data - dmin) / (dmax - dmin + 1e-8)

        if self.transform is not None:
            data = self.transform(data)

        corrupt = data * (torch.rand_like(data) > self.dl_ratio).float()
        return data, corrupt, self.labels[idx]


def create_dataloaders(
    root_dirs: Sequence[str],
    negative_root: Optional[str] = None,
    batch_size: int = 8,
    val_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (32, 32),
    dl_ratio: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    set_seed(seed)
    transform = transforms.Resize(list(image_size))
    full = GestureDataset(
        root_dirs=root_dirs,
        negative_root=negative_root,
        transform=transform,
        dl_ratio=dl_ratio,
    )
    val_size = int(len(full) * val_split)
    train_size = len(full) - val_size
    train_ds, val_ds = random_split(
        full, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
