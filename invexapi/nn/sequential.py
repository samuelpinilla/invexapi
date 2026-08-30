"""``ixSequential``: a stack of invex-certified layers, certified iff every child is."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from ..metadata import Invariant, documented
from .base import Certificate, certify_composition

__all__ = ["ixSequential"]


@documented(
    invariants=[
        Invariant("unlike Sum, ixSequential auto-derives its invex certificate from its children"),
        Invariant("every child must expose `.invex`, or construction raises TypeError"),
    ]
)
class ixSequential(nn.Sequential):
    """A stack of invex-certified layers; certified iff every child is."""

    def __init__(self, *layers: nn.Module) -> None:
        for layer in layers:
            if not hasattr(layer, "invex"):
                raise TypeError(
                    f"ixSequential only accepts children exposing `.invex` "
                    f"(ixLinear/ixConv2d/ixSequential); got {type(layer).__name__}"
                )
        super().__init__(*layers)

    @property
    def invex(self) -> Optional[Certificate]:
        return certify_composition(self)
