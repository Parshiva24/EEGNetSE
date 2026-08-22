"""Reusable network layers: SE ranking blocks and constrained conv/linear.

Two Squeeze-and-Excitation variants implement the paper's *ranking* layers:

* :class:`FilterRankingSE` -- ranks feature maps (channels), the classic SE block.
* :class:`ElectrodeRankingSE` -- ranks electrodes (the height axis) *per* feature
  map. This is a vectorised rewrite of the original per-channel Python loop; the
  arithmetic is identical (the same two FC weights are shared across feature
  maps) but it runs in a single batched matmul instead of a loop over maps.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FilterRankingSE(nn.Module):
    """Squeeze-and-Excitation over feature maps (filter ranking).

    Squeezes each feature map to a scalar by global average pooling, then
    re-weights the maps through a two-layer bottleneck with reduction ``r``.
    """

    def __init__(self, in_channels: int, reduction: int = 3):
        super().__init__()
        hidden = in_channels // reduction
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        #: Mean excitation per feature map from the last forward pass (for analysis).
        self.last_scale: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        s = self.global_avg_pool(x).view(b, c)
        s = self.sigmoid(self.fc2(self.relu(self.fc1(s))))
        self.last_scale = s.mean(dim=0, keepdim=True).detach().cpu()
        return x * s.view(b, c, 1, 1)


class ElectrodeRankingSE(nn.Module):
    """Squeeze-and-Excitation over electrodes, applied per feature map.

    For input ``(b, c, height=electrodes, width=time)`` the temporal axis is
    averaged out, a shared two-layer bottleneck (reduction ``r``) produces one
    weight per electrode, and every feature map is scaled by those weights.
    """

    def __init__(self, n_electrodes: int, reduction: int = 3):
        super().__init__()
        hidden = n_electrodes // reduction
        self.fc1 = nn.Linear(n_electrodes, hidden, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, n_electrodes, bias=False)
        self.sigmoid = nn.Sigmoid()
        #: Mean electrode weights per feature map, shape (c, n_electrodes).
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, _ = x.size()
        s = x.mean(dim=-1)                       # (b, c, h) squeeze over time
        s = s.reshape(b * c, h)
        s = self.sigmoid(self.fc2(self.relu(self.fc1(s))))
        s = s.reshape(b, c, h)
        self.last_weights = s.mean(dim=0).detach().cpu()   # (c, h)
        return x * s.unsqueeze(-1)               # broadcast over time


# --------------------------------------------------------------------------- #
# Weight-norm-constrained layers and temporal aggregators used by FBCNet.
# --------------------------------------------------------------------------- #

class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d whose kernels are renormed to a max L2 norm each forward pass."""

    def __init__(self, *args, do_weight_norm: bool = True, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.do_weight_norm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    """Linear whose weights are renormed to a max L2 norm each forward pass."""

    def __init__(self, *args, do_weight_norm: bool = True, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.do_weight_norm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class LogVarLayer(nn.Module):
    """Log-variance temporal aggregator (FBCNet's default)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6))


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
