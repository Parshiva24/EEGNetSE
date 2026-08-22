"""Generic training and evaluation loops (model-agnostic)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from eegnetse.training.earlystop import EarlyStopping
from eegnetse.training.metrics import accuracy


def _run_eval(model, loader, device, criterion):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets.squeeze()).item()
            total_acc += accuracy(outputs, targets)
    n = len(loader)
    return total_loss / n, total_acc / n


def train(
    model,
    train_loader,
    val_loader,
    config,
    device,
    *,
    warmup_epochs: int = 0,
    min_acc: float = 0.0,
    verbose: bool = False,
):
    """Train ``model`` with Adam + early stopping; restore the best weights.

    Only parameters with ``requires_grad=True`` are optimised, so freezing for
    fine-tuning is controlled entirely by the caller
    (see :func:`eegnetse.training.protocol.apply_finetune_config`).
    """
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate
    )
    criterion = nn.CrossEntropyLoss().to(device)
    stopper = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
        warmup_epochs=warmup_epochs,
        min_acc=min_acc,
    )

    for epoch in range(config.max_epochs):
        model.train()
        running = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets.squeeze())
            loss.backward()
            optimizer.step()
            running += loss.item()

        val_loss, val_acc = _run_eval(model, val_loader, device, criterion)
        if verbose:
            train_loss = running / len(train_loader)
            print(
                f"Epoch {epoch + 1}/{config.max_epochs} | "
                f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.2f}%"
            )
        if stopper.step(epoch, val_acc, model):
            if verbose:
                print(f"Early stop at epoch {epoch + 1} (best val_acc {stopper.best_acc:.2f}%)")
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
    return model


def evaluate(model, loader, device, *, return_scales: bool = False):
    """Return test accuracy (percent); optionally the SE ranking scales too."""
    model.eval()
    correct, total = 0, 0
    scales = None
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            correct += (outputs.argmax(dim=1) == targets.squeeze()).sum().item()
            total += targets.size(0)
            if return_scales and hasattr(model, "ranking_scales"):
                scales = model.ranking_scales()
    acc = 100.0 * correct / total
    return (acc, scales) if return_scales else acc
