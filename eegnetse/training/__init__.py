"""Training loop, metrics and the two-stage transfer-learning protocol."""

from eegnetse.training.tensors import make_loader
from eegnetse.training.metrics import accuracy
from eegnetse.training.earlystop import EarlyStopping
from eegnetse.training.trainer import train, evaluate
from eegnetse.training.protocol import (
    FINETUNE_CONFIGS,
    apply_finetune_config,
    pretrain_base,
    finetune_subject,
    evaluate_online,
)

__all__ = [
    "make_loader",
    "accuracy",
    "EarlyStopping",
    "train",
    "evaluate",
    "FINETUNE_CONFIGS",
    "apply_finetune_config",
    "pretrain_base",
    "finetune_subject",
    "evaluate_online",
]
