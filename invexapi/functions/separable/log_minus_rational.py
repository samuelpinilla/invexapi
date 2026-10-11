from __future__ import annotations

import torch

from ...metadata import DesignDecision, Invariant, Provenance, documented
from ...core import Certificate, Penalty, Reference
from ._admissible import admissible_certificate
from ._scalar_prox import global_prox_magnitude, largest_real_cubic_root

__all__ = ["LogMinusRationalPenalty"]

_INVEXITY_REFERENCE = Reference(
    authors="Pinilla, Mu, Bourne, Thiyagalingam",
    title="Improved Imaging by Invex Regularizers with Global Optima Guarantees",
    venue="NeurIPS",
    year=2022,
    locator="Lemma 1, item 5 (Eq. 10)",
)

# beta at which g_beta is exactly NeurIPS 2022 Eq. (10), the only cited case.
_CITED_BETA = 2.0


@documented(
    provenance=[
        Provenance(
            description=(
                "proximal operator solved in closed form as a root of the cubic "
                "optimality condition (Cardano's formula); beta generalizes the "
                "denominator 2+2|x| of Eq. (10) to beta+beta|x|"
            ),
            reference=_INVEXITY_REFERENCE,
        )
    ],
    decisions=[
        DesignDecision(
            choice="plain differentiable torch ops (ordinary autograd, no custom Function)",
            rejected="a custom torch.autograd.Function with a hand-written analytic backward",
            rationale=(
                "the prox is a single closed-form cubic-root expression with no loop "
                "to unroll, so there's no memory to save by hand-writing a backward, "
                "and the paper gives no closed-form gradient of this prox to "
                "validate a hand-written one against"
            ),
        ),
        DesignDecision(
            choice="return the global minimizer: largest real cubic root compared against z=0",
            rejected="the largest root clamped at 0, as in the paper's reference kernels",
            rationale=(
                "the prox subproblem is nonconvex once lamb*step exceeds 27/4 (beta=2), "
                "where the positive stationary point can lose to z=0; in the convex "
                "regime both agree"
            ),
        ),
        DesignDecision(
            choice="certify invex only at beta=2 and reject beta<1",
            rejected="certifying every beta, or allowing beta in (0,1)",
            rationale=(
                "only beta=2 is Eq. (10); for beta<1 the derivative at 0+ is negative, "
                "so 0 is a kink local maximum and the function is not invex"
            ),
        ),
    ],
    invariants=[
        Invariant(
            "tests/reference_numpy.py::log_minus_rational_prox_np transcribes the "
            "paper's kernel; it must agree with prox() wherever the subproblem is convex."
        )
    ],
    example=lambda: LogMinusRationalPenalty(lamb=0.1),
)
class LogMinusRationalPenalty(Penalty):
    """Log-minus-rational penalty ``g(x) = lamb * sum log(1+|x|) - |x|/(beta+beta|x|)``.

    ``beta=2`` (the default) is the invex regularizer of NeurIPS 2022 Eq. (10).
    """

    def __init__(self, lamb: float, beta: float = _CITED_BETA):
        super().__init__()
        if lamb <= 0.0:
            raise ValueError(f"lamb must be positive, got {lamb}")
        if beta < 1.0:
            raise ValueError(f"beta must be >= 1 (not invex below 1), got {beta}")
        self.lamb = lamb
        self.beta = beta
        if beta == _CITED_BETA:
            self._certify("invex", Certificate(status="assumed", source="manual", reference=_INVEXITY_REFERENCE))
        self._certify("admissible", admissible_certificate("lamb * (log(1 + w) - w/(beta + beta*w)), any beta >= 1"))

    def profile(self, w: torch.Tensor) -> torch.Tensor:
        """Scalar profile s with g(x) = sum s(|x_i|)."""
        return self.lamb * self._phi(w)

    def _phi(self, z: torch.Tensor) -> torch.Tensor:
        return torch.log1p(z) - z / (self.beta * (1.0 + z))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.lamb * self._phi(x.abs()).sum()

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        absx = x.abs()
        return self.lamb * torch.sign(x) * (1.0 / (1.0 + absx) - 1.0 / (self.beta * (1.0 + absx) ** 2))

    def prox(self, x: torch.Tensor, step: float) -> torch.Tensor:
        lamb = self.lamb * step
        beta = self.beta
        t = x.abs()
        sign = torch.sign(x)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)

        # beta*(1+z)^2 * (lamb*phi'(z) + z - t) = 0, expanded.
        b = 2.0 * beta - beta * t
        c = beta * lamb + beta - 2.0 * beta * t
        d = (beta - 1.0) * lamb - beta * t
        root = largest_real_cubic_root(beta, b, c, d)

        return sign * global_prox_magnitude(root, t, lamb, self._phi)

