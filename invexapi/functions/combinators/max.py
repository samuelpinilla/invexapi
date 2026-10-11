from __future__ import annotations

import torch

from ...core import Loss

__all__ = ["Max"]


class Max(Loss):
    """``max(f(x), h(x))``, e.g. a robust objective bounding the worse of two errors.

    ``grad`` is the gradient of the active branch (``f`` on ties) - a valid
    subgradient choice, not differentiable where the branches cross. No prox.
    Certified quasi-invex when both branches are invex (``max_quasi_invex``).
    """

    def __init__(self, f: Loss, h: Loss):
        super().__init__()
        self.f = f
        self.h = h

    @property
    def parts(self) -> tuple:
        return (self.f, self.h)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(self.f.value(x), self.h.value(x))

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return self.f.grad(x) if self.f.value(x) >= self.h.value(x) else self.h.grad(x)
