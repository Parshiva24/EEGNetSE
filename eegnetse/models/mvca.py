"""MVCA baseline (Rong, Yang & Guan, IEEE TNSRE 2024, 32:3399-3409).

Placeholder for the state-of-the-art comparison required by reviewer comments
R1.3 / R2.3 (revision experiment A4). The two-stage training/evaluation harness
in :mod:`eegnetse.training` is model-agnostic, so once the layers below are
filled in, MVCA runs through the identical protocol as every other model.

Left as an explicit stub because architectural fidelity to the source paper
matters and should be implemented deliberately in the experiment phase rather
than guessed here.
"""

from __future__ import annotations

import torch.nn as nn

from eegnetse.config import N_CHANNELS, N_SAMPLES, N_CLASSES


class MVCA(nn.Module):
    """Multi-View Convolutional Attention network -- TO IMPLEMENT (experiment A4)."""

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        n_channels: int = N_CHANNELS,
        n_samples: int = N_SAMPLES,
    ):
        super().__init__()
        raise NotImplementedError(
            "MVCA (Rong et al., TNSRE 2024) is not yet implemented. "
            "Implement per the source paper for revision experiment A4."
        )
