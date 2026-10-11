from __future__ import annotations

from typing import Optional

import torch

from ...core import Loss
from ...metadata import DesignDecision, documented
from ...operators import LinearOperator

__all__ = ["Composed"]


@documented(
    decisions=[
        DesignDecision(
            choice="Composed carries value/grad only, no prox",
            rejected="a generic prox of g(Hx - b)",
            rationale=(
                "the prox of g o H has no closed form unless H H^H is a multiple of the "
                "identity; solvers that need it split on H instead (LinearizedADMM, "
                "FASTA) and use Composed only to read off what is certified"
            ),
        )
    ],
)
class Composed(Loss):
    """``g(H x - b)`` for a function ``g`` and a linear operator ``H``.

    ``grad`` is ``H^H grad_g(H x - b)`` (the adjoint is the conjugate transpose, so
    this is the gradient w.r.t. ``Re<., .>`` for complex ``x`` too). Certificates
    come from :mod:`invexapi.rules` (``convex_affine``, ``affine_invex_construction``,
    ``admissible_difference_1d``, ``admissible_tv_2d``).
    """

    def __init__(self, g: Loss, H: LinearOperator, b: Optional[torch.Tensor] = None):
        super().__init__()
        self.g = g
        self.H = H
        self.b = b

    @property
    def parts(self) -> tuple:
        return (self.g,)

    def _inner(self, x: torch.Tensor) -> torch.Tensor:
        hx = self.H.apply(x)
        return hx if self.b is None else hx - self.b

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.g.value(self._inner(x))

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return self.H.adjoint(self.g.grad(self._inner(x)))
