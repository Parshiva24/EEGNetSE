"""EEGNet-SE: subject-specific deep learning for motor-imagery direction decoding.

A clean, reproducible re-implementation of the code behind the manuscript
*"Subject Specific Deep Learning Model for Motor Imagery Direction Decoding"*.

Three modules only:

    eegnetse.core    model (EEGNet-SE), preprocessing, data loading
    eegnetse.train   base pre-training + per-subject fine-tuning  (python -m eegnetse.train)
    eegnetse.infer   reproduce paper accuracies from checkpoints  (python -m eegnetse.infer)
"""

from eegnetse.core import (
    EEGNet, build_model, set_trainable,
    preprocess, calibration_dataset, online_session, make_loader,
    get_device, set_seed,
)

__all__ = [
    "EEGNet", "build_model", "set_trainable",
    "preprocess", "calibration_dataset", "online_session", "make_loader",
    "get_device", "set_seed",
    "pretrain_base", "finetune_subject", "evaluate_checkpoint",
]

__version__ = "0.2.0"


def __getattr__(name):  # lazy re-exports (avoid importing torch-heavy train/infer eagerly)
    if name in ("pretrain_base", "finetune_subject"):
        from eegnetse import train
        return getattr(train, name)
    if name == "evaluate_checkpoint":
        from eegnetse.infer import evaluate_checkpoint
        return evaluate_checkpoint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
