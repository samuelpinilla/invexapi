"""``ixConv2d``: affine -> activation -> affine with two ``nn.Conv2d`` stages."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from ..metadata import documented
from .base import Certificate, LAYER_REFERENCE, NO_RESTRICTION_DECISION, UNVERIFIED_INVARIANT

__all__ = ["ixConv2d"]


@documented(
    provenance=[],
    decisions=[NO_RESTRICTION_DECISION],
    invariants=[UNVERIFIED_INVARIANT],
)
class ixConv2d(nn.Module):
    """``A2 @ sigma(A1 @ x + b1) + b2`` with two ``nn.Conv2d`` affine stages."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        **conv_kwargs,
    ) -> None:
        super().__init__()
        self.affine1 = nn.Conv2d(in_channels, hidden_channels, kernel_size, **conv_kwargs)
        self.activation = activation
        self.affine2 = nn.Conv2d(hidden_channels, out_channels, kernel_size, **conv_kwargs)
        self._invex = Certificate(status="assumed", source="manual", reference=LAYER_REFERENCE)

    @property
    def invex(self) -> Optional[Certificate]:
        return self._invex

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.affine2(self.activation(self.affine1(x)))
