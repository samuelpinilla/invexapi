from __future__ import annotations

import torch

from ..metadata import documented
from .base import Certificate, Penalty, Reference

__all__ = ["L1Penalty"]

_CONVEXITY_REFERENCE = Reference(
    authors="Boyd, S. and Vandenberghe, L.",
    title="Convex Optimization",
    venue="Cambridge University Press",
    year=2004,
    locator="Section 3.1.2 (norms are convex)",
)


@documented(example=lambda: L1Penalty(lamb=0.1))
class L1Penalty(Penalty):
    """Convex L1 penalty ``g(x) = lamb * ||x||_1``.

    A plain textbook penalty, not from any of the source papers.
    """

    def __init__(self, lamb: float):
        super().__init__()
        if lamb <= 0.0:
            raise ValueError(f"lamb must be positive, got {lamb}")
        self.lamb = lamb

        convexity = Certificate(status="assumed", source="manual", reference=_CONVEXITY_REFERENCE)
        self._certify("convex", convexity)
        self._certify("invex", convexity)
        self._certify("quasi_convex", convexity)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.lamb * x.abs().sum()

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return self.lamb * torch.sign(x)

    def prox(self, x: torch.Tensor, step: float) -> torch.Tensor:
        threshold = self.lamb * step
        return torch.sign(x) * torch.clamp(x.abs() - threshold, min=0.0)
