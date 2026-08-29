"""Model, preprocessing and data loading for EEGNet-SE.

Everything that both training (:mod:`eegnetse.train`) and inference
(:mod:`eegnetse.infer`) need lives here, ported faithfully from the reference
implementation that produced the committed checkpoints in ``trained_model/``:

    * baseline correction (2500 -> 2000 samples) + notch 50 Hz + band-pass
    * ``EEGNet`` with Squeeze-and-Excitation (electrode-SE + filter-SE, r=3),
      a soft-max head and a weight-normalised dense layer.

The channel/sample/parameter names match the checkpoints byte-for-byte, so a
model built here loads them with ``strict=True``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import scipy.io
from scipy import signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Constants (fixed by the acquisition protocol and the trained checkpoints)
# --------------------------------------------------------------------------- #
FS = 500                      # sampling rate (Hz)
N_CHANNELS = 27               # EEG electrodes
N_SAMPLES = 2000              # samples per trial after baseline correction
N_CLASSES = 2                 # left / right motor-imagery direction
BASELINE_SAMPLES = 500        # leading samples used as the baseline window
SE_REDUCTION = 3              # squeeze-and-excitation reduction ratio

BANDPASS = (0.5, 45.0)        # Butterworth band-pass (Hz)
FILT_ORDER = 5
NOTCH_FREQ = 50.0             # mains notch (Hz)
NOTCH_Q = 30.0

CALIB_SUBJECTS = range(1, 8)      # S01-S07: calibration session only (base pre-training)
ONLINE_SUBJECTS = range(8, 21)    # S08-S20: online session (subject-specific evaluation)

_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _REPO_ROOT / "data"                # data/S##_mitrials.mat
MODEL_DIR = _REPO_ROOT / "trained_model"      # committed reference checkpoints


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def baseline_correction(X: np.ndarray, baseline_samples: int = BASELINE_SAMPLES) -> np.ndarray:
    """Subtract each trial's baseline mean, then drop the baseline window.

    ``X`` has shape ``(trials, samples, channels)`` (e.g. 2500 samples); the
    returned array is ``(trials, samples - baseline_samples, channels)``.
    """
    n_trials = X.shape[0]
    out = np.zeros_like(X)
    for t in range(n_trials):
        base = X[t, : baseline_samples - 1, :]
        eeg = X[t, baseline_samples:, :]
        out[t, baseline_samples:, :] = eeg - np.mean(base, axis=0)
    return out[:, baseline_samples:, :]


def bandpass_filtering(X: np.ndarray, fs: int = FS, fcut=BANDPASS, order: int = FILT_ORDER) -> np.ndarray:
    """Zero-phase mains-notch + Butterworth band-pass, per trial and channel."""
    n_trials, _, n_channels = X.shape
    out = np.zeros_like(X)
    b_notch, a_notch = signal.iirnotch(NOTCH_FREQ, NOTCH_Q, fs)
    b, a = signal.butter(order, list(fcut), fs=fs, btype="band", output="ba")
    for t in range(n_trials):
        for c in range(n_channels):
            notched = signal.filtfilt(b_notch, a_notch, X[t, :, c])
            out[t, :, c] = signal.filtfilt(b, a, notched)
    return out


def preprocess(X: np.ndarray, fcut=BANDPASS) -> np.ndarray:
    """Full pipeline: baseline correction then notch + band-pass filtering."""
    return bandpass_filtering(baseline_correction(X), fcut=fcut)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _mat_path(data_dir, subject: int) -> str:
    return os.path.join(str(data_dir), f"S{subject:02d}_mitrials.mat")


def load_mat(data_dir, subject: int):
    """Return ``(Xtrain, Ytrain, Xtest, Ytest)`` for a subject.

    ``Xtest``/``Ytest`` are ``None`` for calibration-only subjects (S01-S07),
    whose stored test arrays are all-NaN.
    """
    m = scipy.io.loadmat(_mat_path(data_dir, subject))
    Xtr, Ytr, Xte, Yte = m["Xtrain"], m["Ytrain"], m["Xtest"], m["Ytest"]
    if np.isnan(Xte).all() or np.isnan(Yte).all():
        return np.asarray(Xtr), np.asarray(Ytr), None, None
    return np.asarray(Xtr), np.asarray(Ytr), np.asarray(Xte), np.asarray(Yte)


def calibration_dataset(data_dir=DATA_DIR, subjects=CALIB_SUBJECTS):
    """Pool the calibration trials of ``subjects`` (default S01-S07)."""
    Xs, Ys = [], []
    for s in subjects:
        Xtr, Ytr, _, _ = load_mat(data_dir, s)
        Xs.append(Xtr)
        Ys.append(Ytr)
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)


def online_session(data_dir, subject: int):
    """Return one online subject's ``(X_calib, Y_calib, X_online, Y_online)``."""
    Xtr, Ytr, Xte, Yte = load_mat(data_dir, subject)
    return Xtr, Ytr, Xte, Yte


def make_loader(X, Y, *, batch_size=32, shuffle=False, fcut=BANDPASS):
    """Preprocess ``(X, Y)`` and wrap them in a ``DataLoader``.

    Tensors are shaped ``(N, 1, channels, samples)`` to match ``EEGNet``.
    """
    Xf = preprocess(X, fcut=fcut)
    xt = torch.tensor(Xf, dtype=torch.float32).unsqueeze(1).permute(0, 1, 3, 2)
    yt = torch.tensor(np.asarray(Y), dtype=torch.long)
    return DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=shuffle)


# --------------------------------------------------------------------------- #
# Squeeze-and-Excitation blocks
# --------------------------------------------------------------------------- #
class FilterSE(nn.Module):
    """Channel-wise (filter) SE recalibration over feature maps."""

    def __init__(self, in_channels: int, reduction: int = SE_REDUCTION):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        s = self.pool(x).view(b, c)
        s = self.sigmoid(self.fc2(self.relu(self.fc1(s)))).view(b, c, 1, 1)
        return x * s.expand_as(x)


class ElectrodeSE(nn.Module):
    """Per-feature-map SE recalibration over the electrode (height) dimension."""

    def __init__(self, height: int, reduction: int = SE_REDUCTION):
        super().__init__()
        self.fc1 = nn.Linear(height, height // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(height // reduction, height, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        out = []
        for i in range(c):
            y = x[:, i, :, :]                       # (b, h, w)
            s = self.sigmoid(self.fc2(self.relu(self.fc1(y.mean(dim=-1)))))  # (b, h)
            out.append(y.unsqueeze(1) * s.view(b, 1, h, 1))
        return torch.cat(out, dim=1)


# --------------------------------------------------------------------------- #
# EEGNet-SE
# --------------------------------------------------------------------------- #
class EEGNet(nn.Module):
    """EEGNet with electrode- and filter-SE, a soft-max head and weight-norm dense.

    Parameter names match the committed checkpoints, so ``load_state_dict`` is
    strict.  ``output_softmax`` reproduces the reference behaviour (soft-max in
    ``forward`` combined with ``CrossEntropyLoss`` during training).
    """

    def __init__(self, nb_classes=N_CLASSES, chans=N_CHANNELS, samples=N_SAMPLES,
                 dropout_rate=0.5, kern_length=64, F1=8, D=2, F2=16, output_softmax=True):
        super().__init__()
        self.output_softmax = output_softmax
        self.dropout = nn.Dropout(dropout_rate)

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kern_length), padding="same", bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.se_electrode1 = ElectrodeSE(chans)
        self.se_electrode2 = ElectrodeSE(chans)
        self.se1 = FilterSE(F1)

        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.se2 = FilterSE(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))

        # Block 2
        self.separableConv = nn.Conv2d(F1 * D, F2, (1, 16), padding="same", bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.se3 = FilterSE(F2)
        self.pool2 = nn.AvgPool2d((1, 8))

        # Classifier (weight_norm reparametrises dense -> weight_g/weight_v)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(F2 * (samples // (4 * 8)), nb_classes)
        self.norm_constraint = nn.utils.weight_norm(self.dense)

    def forward(self, x):
        x = self.se_electrode1(x)
        x = self.batchnorm1(self.conv1(x))
        x = self.se_electrode2(x)
        x = self.se1(x)

        x = F.elu(self.batchnorm2(self.depthwiseConv(x)))
        x = self.dropout(self.pool1(self.se2(x)))

        x = F.elu(self.batchnorm3(self.separableConv(x)))
        x = self.dropout(self.pool2(self.se3(x)))

        x = self.dense(self.flatten(x))
        return F.softmax(x, dim=1) if self.output_softmax else x


def build_model(chans=N_CHANNELS, samples=N_SAMPLES, output_softmax=True, **kwargs) -> EEGNet:
    return EEGNet(chans=chans, samples=samples, output_softmax=output_softmax, **kwargs)


#: Layer groups used to select what fine-tuning updates (everything else frozen).
_GROUPS = {
    "se": ["se_electrode1", "se_electrode2", "se1", "se2", "se3"],
    "dense": ["dense", "norm_constraint"],
    "conv": ["conv1", "depthwiseConv", "separableConv"],
}


def set_trainable(model: EEGNet, groups) -> int:
    """Freeze all parameters, then unfreeze the named ``groups``.

    Returns the number of trainable parameters.
    """
    for p in model.parameters():
        p.requires_grad = False
    for g in groups:
        for attr in _GROUPS[g]:
            module = getattr(model, attr, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
