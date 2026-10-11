from __future__ import annotations

import torch

from ...metadata import DesignDecision, documented
from ...core import Certificate, Penalty, Reference
from ._admissible import admissible_certificate

__all__ = ["TikhonovPenalty"]

_CONVEXITY_REFERENCE = Reference(
    authors="Boyd, S. and Vandenberghe, L.",
    title="Convex Optimization",
    venue="Cambridge University Press",
    year=2004,
    locator="Section 3.1.4",
)


@documented(
    decisions=[
        DesignDecision(
            choice="also certify invex and quasi_convex, reusing the same convexity reference",
            rejected="certifying only convex and leaving invex/quasi_convex unset",
            rationale=(
                "convexity unconditionally implies both invexity (every convex "
                "function is invex - Hanson's invexity definition reduces to "
                "standard convexity when its eta map is the identity) and "
                "quasi-convexity (convex sublevel sets follow directly from "
                "convexity); quasi_invex needs no separate certificate since Loss "
                "already derives it from invex automatically"
            ),
            reference=_CONVEXITY_REFERENCE,
        )
    ],
    example=lambda: TikhonovPenalty(lamb=0.1),
)
class TikhonovPenalty(Penalty):
    """Convex L2 (Tikhonov/ridge) penalty ``g(x) = lamb/2 * ||x||^2``.
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
        self._certify("admissible", admissible_certificate("lamb * w^2 / 2"))
        self._certify(
            "differentiable",
            Certificate(
                status="assumed",
                source="manual",
                reference=Reference(
                    authors=_CONVEXITY_REFERENCE.authors,
                    title=_CONVEXITY_REFERENCE.title,
                    venue=_CONVEXITY_REFERENCE.venue,
                    year=_CONVEXITY_REFERENCE.year,
                    locator="Section 3.1.4 (quadratic functions, gradient lamb*x)",
                ),
            ),
        )

    def profile(self, w: torch.Tensor) -> torch.Tensor:
        """Scalar profile s with g(x) = sum s(|x_i|)."""
        return 0.5 * self.lamb * w * w

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.lamb * torch.sum(x * x)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return self.lamb * x

    def prox(self, x: torch.Tensor, step: float) -> torch.Tensor:
        return x / (1.0 + step * self.lamb)
