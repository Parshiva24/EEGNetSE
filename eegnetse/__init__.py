"""EEGNet-SE: subject-specific deep learning for motor-imagery direction decoding.

A clean, reproducible re-implementation of the code behind the manuscript
*"Subject Specific Deep Learning Model for Motor Imagery Direction Decoding"*
(sensors-4477952).

The public surface is intentionally small:

    from eegnetse import Config, build_model
    from eegnetse.data import calibration_dataset, online_session, preprocess
    from eegnetse.training import pretrain_base, finetune_subject, evaluate

See ``README.md`` for the end-to-end workflow and ``eegnetse.config`` for every
tunable hyper-parameter.
"""

from eegnetse.config import Config, set_seed, get_device

__all__ = ["Config", "set_seed", "get_device", "build_model"]

__version__ = "0.1.0"


def build_model(*args, **kwargs):
    """Lazy proxy for :func:`eegnetse.models.build_model`.

    Imported lazily so that ``import eegnetse`` and the pure-NumPy
    ``eegnetse.data.preprocessing`` path do not require PyTorch.
    """
    from eegnetse.models import build_model as _build_model

    return _build_model(*args, **kwargs)
