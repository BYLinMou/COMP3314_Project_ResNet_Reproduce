"""CIFAR-10 data loading and preprocessing utilities."""

from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _build_transforms(train: bool = True) -> transforms.Compose:
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )


def get_cifar10_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Return training and test dataloaders with paper-aligned augmentation."""

    train_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=_build_transforms(train=True),
    )

    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=_build_transforms(train=False),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, test_loader


__all__ = ["get_cifar10_dataloaders"]
