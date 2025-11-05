"""Training and evaluation utilities for CIFAR ResNet reproduction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


def _accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        preds = output.argmax(dim=1)
        correct = preds.eq(target).sum().item()
    return correct / target.size(0)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch: int = 0,
) -> EpochMetrics:
    """Train the model for a single epoch."""

    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    progress = tqdm(dataloader, desc=f"Epoch {epoch} [train]", leave=False)

    for step, (images, targets) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += outputs.argmax(dim=1).eq(targets).sum().item()
        running_total += batch_size

        progress.set_postfix({
            "loss": running_loss / running_total,
            "acc": running_correct / running_total,
        })

    avg_loss = running_loss / running_total
    avg_acc = running_correct / running_total

    return EpochMetrics(loss=avg_loss, accuracy=avg_acc)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0,
) -> EpochMetrics:
    """Evaluate the model on the validation/test set."""

    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    progress = tqdm(dataloader, desc=f"Epoch {epoch} [eval]", leave=False)

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += outputs.argmax(dim=1).eq(targets).sum().item()
        running_total += batch_size

        progress.set_postfix({
            "loss": running_loss / running_total,
            "acc": running_correct / running_total,
        })

    avg_loss = running_loss / running_total
    avg_acc = running_correct / running_total

    return EpochMetrics(loss=avg_loss, accuracy=avg_acc)


def save_checkpoint(
    state: Dict[str, object],
    checkpoint_dir: str,
    filename: str,
) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    return path


__all__ = ["train_one_epoch", "evaluate", "EpochMetrics", "save_checkpoint"]
