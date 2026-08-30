"""``ixLinear``: affine -> activation -> affine with two ``nn.Linear`` stages."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from ..metadata import documented
from .base import Certificate, LAYER_REFERENCE, NO_RESTRICTION_DECISION, UNVERIFIED_INVARIANT

__all__ = ["ixLinear"]


@documented(
    provenance=[],
    decisions=[NO_RESTRICTION_DECISION],
    invariants=[UNVERIFIED_INVARIANT],
)
class ixLinear(nn.Module):
    """``A2 @ sigma(A1 @ x + b1) + b2`` with two ``nn.Linear`` affine stages."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self.affine1 = nn.Linear(in_features, hidden_features)
        self.activation = activation
        self.affine2 = nn.Linear(hidden_features, out_features)
        self._invex = Certificate(status="assumed", source="manual", reference=LAYER_REFERENCE)

    @property
    def invex(self) -> Optional[Certificate]:
        return self._invex

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.affine2(self.activation(self.affine1(x)))
