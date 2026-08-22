"""Build PyTorch ``DataLoader``s from raw NumPy trials.

Handles preprocessing and the tensor layout each model expects:

* EEGNet / EEGNet-SE: ``(batch, 1, n_channels, n_samples)``
* FBCNet: ``(batch, 1, n_channels, n_samples, n_bands)``
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from eegnetse.config import FS
from eegnetse.data.preprocessing import preprocess, multiband_filter, baseline_correction


def make_loader(
    X: np.ndarray,
    Y: np.ndarray,
    config,
    *,
    shuffle: bool = False,
    multiband: bool = False,
    do_preprocess: bool = True,
    batch_size: int | None = None,
) -> DataLoader:
    """Return a ``DataLoader`` of preprocessed trials.

    Parameters
    ----------
    multiband : bool
        If ``True`` produce the 5-D filter-bank tensor for FBCNet; otherwise the
        4-D single-band tensor for EEGNet.
    do_preprocess : bool
        Set ``False`` if ``X`` is already preprocessed (avoids double filtering).
    """
    if do_preprocess:
        if multiband:
            # baseline correction then filter bank (matches the EEGNet path)
            X = multiband_filter(
                baseline_correction(X), fs=FS, order=config.filter_order,
                notch_freq=config.notch_freq, notch_q=config.notch_q,
            )
        else:
            X = preprocess(
                X, band=config.bandpass, order=config.filter_order,
                notch_freq=config.notch_freq, notch_q=config.notch_q,
            )

    # ascontiguousarray: filtfilt/transpose can yield negative strides, which
    # torch.tensor cannot consume.
    x = torch.tensor(np.ascontiguousarray(X), dtype=torch.float32).unsqueeze(1)
    if not multiband:
        # (N, 1, n_samples, n_channels) -> (N, 1, n_channels, n_samples)
        x = x.permute(0, 1, 3, 2).contiguous()

    y = torch.tensor(np.asarray(Y), dtype=torch.long)
    dataset = TensorDataset(x, y)
    return DataLoader(
        dataset,
        batch_size=batch_size or config.batch_size,
        shuffle=shuffle,
    )
