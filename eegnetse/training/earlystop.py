"""Early stopping on validation accuracy.

Generalises the two stopping policies that were tangled together in the original
notebooks:

* **fine-tuning** -- stop after ``patience`` epochs without improvement
  (``warmup_epochs=0``, ``min_acc=0``);
* **base pre-training** -- ignore stopping until at least ``warmup_epochs`` have
  elapsed *and* the best accuracy exceeds ``min_acc`` (the original
  ``epoch > 100`` / ``best_val_acc > 70`` guard).
"""

from __future__ import annotations

import copy


class EarlyStopping:
    def __init__(
        self,
        patience: int = 30,
        min_delta: float = 1e-3,
        warmup_epochs: int = 0,
        min_acc: float = 0.0,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.min_acc = min_acc

        self.best_acc = 0.0
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def step(self, epoch: int, val_acc: float, model) -> bool:
        """Update state with this epoch's accuracy; return ``True`` to stop.

        Snapshots the best model weights internally (retrievable via
        :attr:`best_state`).
        """
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1

        past_warmup = epoch >= self.warmup_epochs
        cleared_floor = self.best_acc > self.min_acc
        if self.counter >= self.patience and past_warmup and cleared_floor:
            self.should_stop = True
        return self.should_stop
