"""Central configuration, reproducibility and device helpers.

Every hyper-parameter that used to be hard-coded (and often *inconsistently*
duplicated) across the notebooks now lives here as a single dataclass, so an
experiment is fully described by one ``Config`` instance.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Dataset layout constants (properties of the recorded corpus, not tunables).
# --------------------------------------------------------------------------- #

#: Subjects S01-S07 provide a calibration session only (used to pre-train the
#: subject-independent base model).
CALIBRATION_ONLY_SUBJECTS = tuple(range(1, 8))

#: Subjects S08-S20 additionally provide an online (feedback) session that is
#: used exclusively for evaluation.
ONLINE_SUBJECTS = tuple(range(8, 21))

N_CHANNELS = 27           # EEG electrodes
FS = 500                  # sampling rate (Hz)
BASELINE_SAMPLES = 500    # 1 s pre-cue baseline that is subtracted then dropped
N_SAMPLES = 2000          # samples per trial fed to the model (4 s @ 500 Hz)
N_CLASSES = 2             # left / right MI direction


@dataclass
class Config:
    """All knobs for a single training/evaluation run.

    The defaults reproduce the settings used by the original notebooks; override
    fields to explore alternatives without touching the library code.
    """

    # -- Paths ------------------------------------------------------------- #
    #: Directory holding ``S01_mitrials.mat`` ... ``S20_mitrials.mat``.
    data_dir: Path = field(default_factory=lambda: _default_data_dir())
    #: Directory where checkpoints and result tables are written.
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    # -- Reproducibility --------------------------------------------------- #
    seed: int = 0

    # -- Preprocessing ----------------------------------------------------- #
    bandpass: tuple[float, float] = (0.5, 45.0)
    filter_order: int = 5
    notch_freq: float = 50.0
    notch_q: float = 30.0

    # -- Model ------------------------------------------------------------- #
    #: ``"eegnet"`` (no ranking), ``"eegnetse"`` (proposed), ``"fbcnet"``, ``"mvca"``.
    model: str = "eegnetse"
    se_reduction: int = 3          # reduction rate r of every SE ranking block
    dropout_rate: float = 0.5
    kernel_length: int = 64
    F1: int = 8
    D: int = 2
    F2: int = 16

    # -- Optimisation ------------------------------------------------------ #
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 1000
    patience: int = 30             # early-stopping patience (epochs)
    min_delta: float = 1e-3        # min val-accuracy gain counted as improvement
    val_ratio: float = 0.1         # held-out fraction for early stopping
    split_seed: int = 42           # RNG for the train/val split

    #: During base pre-training, early stopping is disabled until this many
    #: epochs have elapsed and the best accuracy clears ``pretrain_min_acc``.
    #: (Reproduces the ``epoch > 100`` / ``best_val_acc > 70`` guard.)
    pretrain_warmup_epochs: int = 100
    pretrain_min_acc: float = 70.0

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)

    # Convenience ---------------------------------------------------------- #
    def as_dict(self) -> dict:
        d = asdict(self)
        d["data_dir"] = str(self.data_dir)
        d["output_dir"] = str(self.output_dir)
        return d


def _default_data_dir() -> Path:
    """Resolve the data directory, honouring the ``EEGNETSE_DATA`` env var.

    Falls back to ``<repo-root>/data`` so the package works out of the box when
    the ``.mat`` files sit next to it.
    """
    env = os.environ.get("EEGNETSE_DATA")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "data"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if available) PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is optional for the pure-NumPy preprocessing path
        pass


def get_device():
    """Return the best available device: CUDA, then Apple MPS, then CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
