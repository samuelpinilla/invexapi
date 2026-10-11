from __future__ import annotations

import torch

from ...metadata import DesignDecision, Invariant, Provenance, documented
from ...core import Certificate, Penalty, Reference
from ._admissible import admissible_certificate
from ._scalar_prox import global_prox_magnitude, largest_real_quadratic_root

__all__ = ["LogPenalty"]

_INVEXITY_REFERENCE = Reference(
    authors="Pinilla, Mu, Bourne, Thiyagalingam",
    title="Improved Imaging by Invex Regularizers with Global Optima Guarantees",
    venue="NeurIPS",
    year=2022,
    locator="Lemma 1, item 2 (Eq. 7)",
)

# beta at which lamb*log(1+beta|x|) is exactly NeurIPS 2022 Eq. (7) up to scale.
_CITED_BETA = 1.0


@documented(
    provenance=[
        Provenance(
            description=(
                "proximal operator solved in closed form as the larger root of the "
                "quadratic optimality condition"
            ),
            reference=_INVEXITY_REFERENCE,
        )
    ],
    decisions=[
        DesignDecision(
            choice="parameterize as lamb*log(1+beta|x|)",
            rejected="(lamb/beta)*log(1+beta|x|), the scaling used by the paper's experiment code",
            rationale=(
                "lamb then weights the penalty exactly as in every other Penalty here; "
                "the two differ only by renaming lamb"
            ),
        ),
        DesignDecision(
            choice="return the global minimizer: larger quadratic root compared against z=0",
            rejected="the larger root clamped at 0, as in the paper's experiment code",
            rationale=(
                "the prox subproblem is nonconvex once lamb*step*beta^2 > 1, where the "
                "positive stationary point can lose to z=0; in the convex regime both agree"
            ),
        ),
        DesignDecision(
            choice="certify invex only at beta=1",
            rejected="certifying every beta>0",
            rationale=(
                "beta=1 is Eq. (7) verbatim; other beta are invex by positive rescaling "
                "of the argument, but that argument is not in the cited reference"
            ),
        ),
    ],
    invariants=[
        Invariant(
            "tests/reference_numpy.py::log_prox_np transcribes the paper's prox; it must "
            "agree with prox(lamb=lamb_ref/beta) wherever the subproblem is convex."
        )
    ],
    example=lambda: LogPenalty(lamb=0.1),
)
class LogPenalty(Penalty):
    """Log penalty ``g(x) = lamb * sum log(1 + beta*|x|)``.

    ``beta=1`` (the default) is the invex regularizer of NeurIPS 2022 Eq. (7).
    """

    def __init__(self, lamb: float, beta: float = _CITED_BETA):
        super().__init__()
        if lamb <= 0.0:
            raise ValueError(f"lamb must be positive, got {lamb}")
        if beta <= 0.0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.lamb = lamb
        self.beta = beta
        if beta == _CITED_BETA:
            self._certify("invex", Certificate(status="assumed", source="manual", reference=_INVEXITY_REFERENCE))
        self._certify("admissible", admissible_certificate("lamb * log(1 + beta*w), any beta > 0"))

    def profile(self, w: torch.Tensor) -> torch.Tensor:
        """Scalar profile s with g(x) = sum s(|x_i|)."""
        return self.lamb * self._phi(w)

    def _phi(self, z: torch.Tensor) -> torch.Tensor:
        return torch.log1p(self.beta * z)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.lamb * self._phi(x.abs()).sum()

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return self.lamb * self.beta * torch.sign(x) / (1.0 + self.beta * x.abs())

    def prox(self, x: torch.Tensor, step: float) -> torch.Tensor:
        lamb = self.lamb * step
        beta = self.beta
        t = x.abs()
        sign = torch.sign(x)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)

        # (1+beta*z) * (lamb*phi'(z) + z - t) = 0, expanded.
        root = largest_real_quadratic_root(beta, 1.0 - beta * t, lamb * beta - t)

        return sign * global_prox_magnitude(root, t, lamb, self._phi)
