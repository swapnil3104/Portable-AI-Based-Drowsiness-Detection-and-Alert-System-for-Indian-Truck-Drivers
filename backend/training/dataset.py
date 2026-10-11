"""Dataset and DataLoader utilities for Eye-State Classification in DrishtiDrive AI.

Provides dataset loading, data augmentation, preprocessing, and DataLoader
generation for train, validation, and held-out test splits.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


CLASS_NAMES = ["open", "closed"]
CLASS_TO_IDX = {"open": 0, "closed": 1}


class EyeDataset(Dataset):
    """PyTorch Dataset loading eye images from a split folder."""

    def __init__(self, split_dir: str | Path, transform: Optional[transforms.Compose] = None) -> None:
        """Initialize dataset by indexing all valid image files in split_dir."""
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        for class_name, class_idx in CLASS_TO_IDX.items():
            class_folder = self.split_dir / class_name
            if class_folder.exists():
                for file_path in sorted(class_folder.iterdir()):
                    if file_path.is_file() and file_path.suffix.lower() in valid_exts:
                        self.samples.append((file_path, class_idx))

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load, transform, and return single image and its integer label."""
        file_path, label = self.samples[idx]
        image = Image.open(file_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def get_class_counts(self) -> Dict[str, int]:
        """Compute sample counts per class."""
        counts = {name: 0 for name in CLASS_NAMES}
        for _, label in self.samples:
            counts[CLASS_NAMES[label]] += 1
        return counts


def get_transforms(image_size: int = 96) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return train and validation/test torchvision transform pipelines."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


def get_dataloaders(
    data_dir: str | Path,
    image_size: int = 96,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Create train, validation, and test DataLoaders for eye state dataset."""
    root = Path(data_dir)
    train_t, eval_t = get_transforms(image_size)

    train_ds = EyeDataset(root / "train", transform=train_t)
    val_ds = EyeDataset(root / "val", transform=eval_t)
    test_ds = EyeDataset(root / "test", transform=eval_t)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )

    split_counts = {
        "train": len(train_ds),
        "val": len(val_ds),
        "test": len(test_ds),
    }
    return train_loader, val_loader, test_loader, split_counts
