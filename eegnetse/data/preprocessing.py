"""Signal preprocessing for the MI direction-decoding pipeline.

All routines operate on arrays shaped ``(n_trials, n_samples, n_channels)`` --
the layout of the raw ``.mat`` trials -- and are pure NumPy/SciPy so they can be
unit-tested without a GPU or PyTorch.

The single :func:`preprocess` entry point (baseline correction -> notch ->
band-pass) replaces the copy that was duplicated, with subtle differences, in
every notebook.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from eegnetse.config import BASELINE_SAMPLES, FS


def baseline_correction(X: np.ndarray, baseline_samples: int = BASELINE_SAMPLES) -> np.ndarray:
    """Subtract the pre-cue baseline mean, then drop the baseline segment.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_samples, n_channels)
        Raw trials. The first ``baseline_samples`` samples are the baseline.

    Returns
    -------
    ndarray, shape (n_trials, n_samples - baseline_samples, n_channels)
        Baseline-corrected trials with the baseline segment removed.
    """
    baseline_mean = X[:, : baseline_samples - 1, :].mean(axis=1, keepdims=True)
    return X[:, baseline_samples:, :] - baseline_mean


def bandpass_filter(
    X: np.ndarray,
    fs: int = FS,
    band: tuple[float, float] = (0.5, 45.0),
    order: int = 5,
    notch_freq: float | None = 50.0,
    notch_q: float = 30.0,
) -> np.ndarray:
    """Zero-phase notch + Butterworth band-pass filtering.

    Applies an optional line-noise notch followed by a band-pass, both with
    :func:`scipy.signal.filtfilt` (zero phase). Filtering is vectorised over
    trials and channels via the ``axis`` argument instead of the original
    double Python loop.

    Parameters
    ----------
    notch_freq : float or None
        Line-noise frequency to notch out; pass ``None`` to skip the notch.
    """
    X = np.asarray(X, dtype=float)
    out = X
    if notch_freq is not None:
        b_notch, a_notch = signal.iirnotch(notch_freq, notch_q, fs)
        out = signal.filtfilt(b_notch, a_notch, out, axis=1)
    b, a = signal.butter(order, band, fs=fs, btype="band", output="ba")
    out = signal.filtfilt(b, a, out, axis=1)
    return out


def multiband_filter(
    X: np.ndarray,
    fs: int = FS,
    bands: tuple[tuple[float, float], ...] = (
        (4, 8), (8, 12), (12, 16), (16, 20), (20, 24),
        (24, 28), (28, 32), (32, 36), (36, 40),
    ),
    order: int = 5,
    notch_freq: float | None = 50.0,
    notch_q: float = 30.0,
) -> np.ndarray:
    """Filter into a bank of frequency bands (input format expected by FBCNet).

    Returns
    -------
    ndarray, shape (n_trials, n_channels, n_samples, n_bands)
    """
    filtered = [
        bandpass_filter(X, fs=fs, band=band, order=order,
                        notch_freq=notch_freq, notch_q=notch_q)
        for band in bands
    ]
    # stack -> (n_trials, n_samples, n_channels, n_bands) then to (t, c, s, b)
    Xout = np.stack(filtered, axis=-1)
    return np.transpose(Xout, (0, 2, 1, 3))


def preprocess(
    X: np.ndarray,
    fs: int = FS,
    band: tuple[float, float] = (0.5, 45.0),
    order: int = 5,
    notch_freq: float | None = 50.0,
    notch_q: float = 30.0,
    baseline_samples: int = BASELINE_SAMPLES,
) -> np.ndarray:
    """Full single-band pipeline: baseline correction then notch + band-pass."""
    X = baseline_correction(X, baseline_samples=baseline_samples)
    return bandpass_filter(
        X, fs=fs, band=band, order=order, notch_freq=notch_freq, notch_q=notch_q
    )


def euclidean_alignment(X_target: np.ndarray, X_source: np.ndarray) -> np.ndarray:
    """Euclidean Alignment (EA) of ``X_source`` onto the ``X_target`` reference.

    Whitens the source trials by the inverse square root of the target's average
    spatial covariance, reducing cross-session/subject covariate shift.

    Parameters
    ----------
    X_target, X_source : ndarray, shape (n_trials, n_channels, n_samples)

    Returns
    -------
    ndarray
        Aligned copy of ``X_source`` (same shape).
    """
    X_target = np.asarray(X_target).squeeze()
    X_source = np.asarray(X_source).squeeze()
    if X_target.size == 0:
        X_target = X_source

    n_trials = X_source.shape[0]
    n_channels = X_source.shape[1]

    cov_target = np.zeros((n_channels, n_channels))
    for trial in range(X_target.shape[0]):
        cov_target += np.cov(X_target[trial])
    cov_target /= X_target.shape[0]

    eigvals, eigvecs = np.linalg.eigh(cov_target)
    r_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    aligned = np.zeros_like(X_source)
    for trial in range(n_trials):
        aligned[trial] = r_inv_sqrt @ X_source[trial]
    return aligned
