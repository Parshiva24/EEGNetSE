"""Evaluation metrics."""

from __future__ import annotations

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy in percent for a batch of ``logits`` vs. ``targets``."""
    preds = logits.argmax(dim=1)
    correct = (preds == targets.squeeze()).sum().item()
    return 100.0 * correct / targets.size(0)
