from __future__ import annotations

import torch

from ..metadata import DesignDecision, Provenance, documented
from .base import Certificate, Penalty, Reference

__all__ = ["LogInvexPenalty"]

_INVEXITY_REFERENCE = Reference(
    authors="Pinilla, Mu, Bourne, Thiyagalingam",
    title="Improved Imaging by Invex Regularizers with Global Optima Guarantees",
    venue="NeurIPS",
    year=2022,
    locator="Lemma 1, item 5 (Eq. 10)",
)


def _cbrt(v: torch.Tensor) -> torch.Tensor:
    """Real cube root, valid for negative inputs too (C's ``cbrtf`` semantics)."""
    return torch.sign(v) * v.abs().pow(1.0 / 3.0)


@documented(
    provenance=[
        Provenance(
            files=("codes from conference paper Improved Imaging by Invex Regularizers" 
                   "with Global Optima Guarantees",),
            description="Cardano cubic-root prox solve",
        )
    ],
    decisions=[
        DesignDecision(
            choice="plain differentiable torch ops (ordinary autograd, no custom Function)",
            rejected="a custom torch.autograd.Function with a hand-written analytic backward",
            rationale=(
                "the prox is a single closed-form cubic-root expression with no loop "
                "to unroll, so there's no memory to save by hand-writing a backward, "
                "and no analytic-gradient kernel exists in the source to validate an "
                "alternative against "
            ),
        )
    ],
    example=lambda: LogInvexPenalty(lamb=0.1),
)
class LogInvexPenalty(Penalty):
    """Invex penalty ``g(x) = log(1+|x|) - |x|/(2+2|x|)``.

    Only takes ``lamb`` — this penalty's shape is fixed (no ``q``-like parameter).
    """

    def __init__(self, lamb: float):
        super().__init__()
        if lamb <= 0.0:
            raise ValueError(f"lamb must be positive, got {lamb}")
        self.lamb = lamb
        self._certify("invex", Certificate(status="assumed", source="manual", reference=_INVEXITY_REFERENCE))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        absx = x.abs()
        return (torch.log1p(absx) - absx / (2.0 + 2.0 * absx)).sum()

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        absx = x.abs()
        sign = torch.sign(x)
        return sign * (1.0 / (1.0 + absx) - 1.0 / (2.0 * (1.0 + absx) ** 2))

    def prox(self, x: torch.Tensor, step: float) -> torch.Tensor:
        lamb = self.lamb * step
        t = x.abs()
        sign = torch.sign(x)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)

        a = 2.0
        b = 4.0 - 2.0 * t
        c = 2.0 * lamb + 2.0 - 4.0 * t
        d = lamb - 2.0 * t

        p = -b / (3.0 * a)
        qq = p ** 3.0 + (b * c - 3.0 * a * d) / (6.0 * a ** 2.0)
        r = c / (3.0 * a)

        aux = torch.clamp(qq ** 2 + (r - p ** 2) ** 3, min=0.0).sqrt()
        beta = torch.clamp(_cbrt(qq + aux) + _cbrt(qq - aux) + p, min=0.0)

        return sign * beta
