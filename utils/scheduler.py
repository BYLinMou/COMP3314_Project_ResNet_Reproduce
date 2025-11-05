"""Learning rate scheduling utilities."""

from __future__ import annotations

from bisect import bisect_right
from typing import Iterable, Sequence

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupMultiStepLR(_LRScheduler):
    """Multi-step LR scheduler with optional warmup in *iterations*.

    Parameters
    ----------
    optimizer:
        Optimizer whose learning rate will be scheduled.
    milestones:
        Sorted increasing sequence of iteration indices where the learning
        rate is decayed by ``gamma``.
    gamma:
        Multiplicative factor of learning rate decay at each milestone.
    warmup_iters:
        Number of initial iterations used for linear warmup.
    warmup_factor:
        Starting factor for warmup (final factor is 1.0).
    last_epoch:
        The index of last iteration. Use -1 to start from the beginning.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: Sequence[int],
        gamma: float = 0.1,
        warmup_iters: int = 0,
        warmup_factor: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        if not list(milestones) == sorted(milestones):
            raise ValueError("Milestones should be a non-decreasing sequence")
        if warmup_iters < 0:
            raise ValueError("warmup_iters must be >= 0")

        self.milestones = list(milestones)
        self.gamma = gamma
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):  # type: ignore[override]
        warmup_multiplier = 1.0
        if self.last_epoch < self.warmup_iters and self.warmup_iters > 0:
            alpha = float(self.last_epoch + 1) / float(self.warmup_iters)
            warmup_multiplier = self.warmup_factor + (1.0 - self.warmup_factor) * alpha

        decay_multiplier = self.gamma ** bisect_right(self.milestones, self.last_epoch)

        return [
            base_lr * warmup_multiplier * decay_multiplier for base_lr in self.base_lrs
        ]


def build_resnet_cifar_scheduler(
    optimizer: Optimizer,
    steps_per_epoch: int,
    epochs: int,
    gamma: float = 0.1,
    warmup_epochs: float = 0.0,
) -> WarmupMultiStepLR:
    """Return scheduler matching the CIFAR ResNet training recipe.

    The original paper decays the learning rate by 10× at 32k and 48k
    iterations (batch updates). We convert these iteration counts into
    scheduler milestones.
    """

    milestone_iters = [32000, 48000]
    total_iters = steps_per_epoch * epochs
    milestones = [m for m in milestone_iters if m < total_iters]

    warmup_iters = int(warmup_epochs * steps_per_epoch)

    return WarmupMultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=gamma,
        warmup_iters=warmup_iters,
    )


__all__ = ["WarmupMultiStepLR", "build_resnet_cifar_scheduler"]
