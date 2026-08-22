"""FBCNet baseline.

Filter-Bank Convolutional Network (Mane et al.): a spatial convolution per
frequency band followed by a temporal variance aggregator. Expects multiband
input shaped ``(batch, 1, n_channels, n_samples, n_bands)`` -- see
:func:`eegnetse.data.multiband_filter`.

Returns **logits** (the original returned log-softmax + used NLL implicitly);
pair with ``CrossEntropyLoss`` for training.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from eegnetse.config import N_CHANNELS, N_CLASSES
from eegnetse.models.layers import Conv2dWithConstraint, LinearWithConstraint, LogVarLayer, Swish


class FBCNet(nn.Module):
    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_classes: int = N_CLASSES,
        n_bands: int = 9,
        m: int = 32,
        stride_factor: int = 4,
        do_weight_norm: bool = True,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.m = m
        self.stride_factor = stride_factor

        # Spatial convolution block: one grouped conv per band.
        self.scb = nn.Sequential(
            Conv2dWithConstraint(
                n_bands, m * n_bands, (n_channels, 1), groups=n_bands,
                max_norm=2, do_weight_norm=do_weight_norm, padding=0,
            ),
            nn.BatchNorm2d(m * n_bands),
            Swish(),
        )
        self.temporal_layer = LogVarLayer(dim=3)
        self.last_layer = LinearWithConstraint(
            m * n_bands * stride_factor, n_classes, max_norm=0.5, do_weight_norm=do_weight_norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (b, 1, chan, time, bands) -> (b, bands, chan, time)
        x = torch.squeeze(x.permute(0, 4, 2, 3, 1), dim=4)
        x = self.scb(x)
        x = x.reshape(*x.shape[0:2], self.stride_factor, x.shape[3] // self.stride_factor)
        x = self.temporal_layer(x)
        x = torch.flatten(x, start_dim=1)
        return self.last_layer(x)
