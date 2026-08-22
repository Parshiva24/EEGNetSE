"""Model definitions and a config-driven factory."""

from __future__ import annotations

from eegnetse.models.eegnet import EEGNet
from eegnetse.models.fbcnet import FBCNet
from eegnetse.models.mvca import MVCA

__all__ = ["EEGNet", "FBCNet", "MVCA", "build_model"]


def build_model(config, **overrides):
    """Instantiate the model named by ``config.model``.

    ``overrides`` are forwarded to the model constructor, letting callers flip
    e.g. ``use_filter_se`` for an ablation without mutating ``config``.
    """
    name = config.model.lower()
    if name in {"eegnet", "eegnetse"}:
        use_se = name == "eegnetse"
        kwargs = dict(
            dropout_rate=config.dropout_rate,
            kernel_length=config.kernel_length,
            F1=config.F1,
            D=config.D,
            F2=config.F2,
            se_reduction=config.se_reduction,
            use_electrode_se=use_se,
            use_filter_se=use_se,
        )
        kwargs.update(overrides)
        return EEGNet(**kwargs)
    if name == "fbcnet":
        return FBCNet(**overrides)
    if name == "mvca":
        return MVCA(**overrides)
    raise ValueError(f"Unknown model '{config.model}'.")
