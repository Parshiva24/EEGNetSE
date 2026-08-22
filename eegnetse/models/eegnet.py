"""EEGNet and the proposed EEGNet-SE (with electrode/filter ranking layers).

A single :class:`EEGNet` class covers the whole family through two flags:

======================  =================  ================
Model                   use_electrode_se   use_filter_se
======================  =================  ================
EEGNet (baseline)       False              False
proposed w/o ranking    False              False   (== baseline)
EEGNet-SE (proposed)    True               True
======================  =================  ================

Intermediate variants (electrode-only / filter-only) are available for
ablations by toggling one flag.

.. note::
   The network returns **logits**. The original notebooks applied ``softmax``
   inside ``forward`` *and* trained with ``CrossEntropyLoss`` (which applies
   log-softmax again); pairing raw logits with ``CrossEntropyLoss`` is the
   correct formulation and is the one deviation from the original numerics.
   Argmax predictions -- and therefore every reported accuracy -- are
   unaffected by this change. See ``CHANGES.md``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from eegnetse.config import N_CHANNELS, N_SAMPLES, N_CLASSES
from eegnetse.models.layers import ElectrodeRankingSE, FilterRankingSE


class EEGNet(nn.Module):
    """EEGNet backbone with optional electrode- and filter-ranking SE blocks."""

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        n_channels: int = N_CHANNELS,
        n_samples: int = N_SAMPLES,
        dropout_rate: float = 0.5,
        kernel_length: int = 64,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        se_reduction: int = 3,
        use_electrode_se: bool = True,
        use_filter_se: bool = True,
        dropout_type: str = "Dropout",
    ):
        super().__init__()
        self.use_electrode_se = use_electrode_se
        self.use_filter_se = use_filter_se

        if dropout_type == "SpatialDropout2D":
            self.dropout = nn.Dropout2d(dropout_rate)
        elif dropout_type == "Dropout":
            self.dropout = nn.Dropout(dropout_rate)
        else:
            raise ValueError("dropout_type must be 'SpatialDropout2D' or 'Dropout'.")

        # --- Block 1: temporal conv -------------------------------------- #
        if use_electrode_se:
            self.se_electrode1 = ElectrodeRankingSE(n_channels, se_reduction)
            self.se_electrode2 = ElectrodeRankingSE(n_channels, se_reduction)
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding="same", bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        if use_filter_se:
            self.se1 = FilterRankingSE(F1, se_reduction)

        # --- Depthwise spatial conv -------------------------------------- #
        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        if use_filter_se:
            self.se2 = FilterRankingSE(F1 * D, se_reduction)
        self.pool1 = nn.AvgPool2d((1, 4))

        # --- Block 2: separable conv ------------------------------------- #
        self.separable_conv = nn.Conv2d(F1 * D, F2, (1, 16), padding="same", bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        if use_filter_se:
            self.se3 = FilterRankingSE(F2, se_reduction)
        self.pool2 = nn.AvgPool2d((1, 8))

        # --- Classifier -------------------------------------------------- #
        self.flatten = nn.Flatten()
        dense = nn.Linear(F2 * (n_samples // (4 * 8)), n_classes)
        self.dense = nn.utils.weight_norm(dense)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        if self.use_electrode_se:
            x = self.se_electrode1(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        if self.use_electrode_se:
            x = self.se_electrode2(x)
        if self.use_filter_se:
            x = self.se1(x)

        x = self.depthwise_conv(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        if self.use_filter_se:
            x = self.se2(x)
        x = self.pool1(x)
        x = self.dropout(x)

        # Block 2
        x = self.separable_conv(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        if self.use_filter_se:
            x = self.se3(x)
        x = self.pool2(x)
        x = self.dropout(x)

        # Classifier (logits)
        x = self.flatten(x)
        return self.dense(x)

    # -- Analysis helpers -------------------------------------------------- #
    def ranking_scales(self) -> dict:
        """Return the most recent SE excitations/weights for interpretation.

        Populated after a forward pass; keys are present only for the SE blocks
        that exist in this configuration.
        """
        scales: dict = {}
        if self.use_electrode_se:
            scales["electrodes_se1"] = self.se_electrode1.last_weights
            scales["electrodes_se2"] = self.se_electrode2.last_weights
        if self.use_filter_se:
            scales["filter_se1"] = self.se1.last_scale
            scales["filter_se2"] = self.se2.last_scale
            scales["filter_se3"] = self.se3.last_scale
        return scales
