"""Robust data-fidelity losses that ICLR 2024 Thm 5 names as admissible.

Each is ``f(x) = sum_i s(x_i)`` with ``s`` a smooth function of ``x_i^2``, written
exactly as in the paper. They are losses, not penalties: value/grad only, no prox
(the source experiments use them only as training losses).
"""

from __future__ import annotations

import math

import torch

from ...certificate import Certificate
from ...core import Loss
from ...metadata import DesignDecision, Reference, documented
from ..separable._admissible import ADMISSIBLE_DEFINITION

__all__ = [
    "CauchyLoss",
    "GemanMcClureLoss",
    "WelschLoss",
    "AdaptiveRobustLoss",
    "SquaredLogMinusRationalLoss",
]


def _theorem5(equation: str, extra: str = "") -> Reference:
    return Reference(**ADMISSIBLE_DEFINITION, locator=f"Theorem 5, Eq. ({equation}){extra}")


class _RobustLoss(Loss):
    """Shared shape: ``value = sum s(x)``, ``profile = s`` on ``w >= 0``."""

    _equation = ""

    def _certify_from_theorem5(self, admissible: bool = True) -> None:
        if admissible:
            self._certify(
                "admissible",
                Certificate(status="assumed", source="manual", reference=_theorem5(self._equation)),
            )
        self._certify(
            "differentiable",
            Certificate(
                status="assumed",
                source="manual",
                reference=_theorem5(self._equation, "; C-infinity by inspection (smooth in x^2)"),
            ),
        )

    def _s(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self._s(x).sum()

    def profile(self, w: torch.Tensor) -> torch.Tensor:
        """Scalar profile s with f(x) = sum s(|x_i|) (ICLR 2024 Def. 4)."""
        return self._s(w)


def _check_positive(name: str, value: float) -> None:
    if not value > 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


@documented(example=lambda: CauchyLoss(delta=1.0))
class CauchyLoss(_RobustLoss):
    """Cauchy loss ``f(x) = sum log(1 + x^2 / delta^2)`` (ICLR 2024 Eq. (4))."""

    _equation = "4"

    def __init__(self, delta: float = 1.0):
        super().__init__()
        _check_positive("delta", delta)
        self.delta = delta
        self._certify_from_theorem5()

    def _s(self, x):
        return torch.log1p(x * x / self.delta ** 2)

    def grad(self, x):
        return 2.0 * x / (self.delta ** 2 + x * x)


@documented(example=lambda: GemanMcClureLoss(delta=1.0))
class GemanMcClureLoss(_RobustLoss):
    """Geman-McClure loss ``f(x) = sum 2x^2 / (x^2 + 4 delta^2)`` (ICLR 2024 Eq. (5))."""

    _equation = "5"

    def __init__(self, delta: float = 1.0):
        super().__init__()
        _check_positive("delta", delta)
        self.delta = delta
        self._certify_from_theorem5()

    def _s(self, x):
        return 2.0 * x * x / (x * x + 4.0 * self.delta ** 2)

    def grad(self, x):
        a = 4.0 * self.delta ** 2
        return 4.0 * a * x / (x * x + a) ** 2


@documented(example=lambda: WelschLoss(delta=1.0))
class WelschLoss(_RobustLoss):
    """Welsch loss ``f(x) = sum 1 - exp(-x^2 / (2 delta^2))`` (ICLR 2024 Eq. (6)).

    Named after Holland & Welsch's M-estimator; the reference spells it "Welsh".
    """

    _equation = "6"

    def __init__(self, delta: float = 1.0):
        super().__init__()
        _check_positive("delta", delta)
        self.delta = delta
        self._certify_from_theorem5()

    def _s(self, x):
        return -torch.expm1(-x * x / (2.0 * self.delta ** 2))

    def grad(self, x):
        return x / self.delta ** 2 * torch.exp(-x * x / (2.0 * self.delta ** 2))


@documented(
    decisions=[
        DesignDecision(
            choice="certify admissible only for alpha <= 2",
            rejected="certifying every real alpha, as Theorem 5 states",
            rationale=(
                "for alpha > 2, s grows faster than w^2, so s(w)/w^2 increases and "
                "Definition 4 fails (confirmed numerically); the paper's own code "
                "restricts alpha to (0, 1.99)"
            ),
        ),
        DesignDecision(
            choice="exact limit formulas at alpha = 2, 0 and -inf",
            rejected="rejecting those alphas (the general formula is 0/0 there)",
            rationale=(
                "they are the quadratic, Cauchy-like log and Welsch-like exp cases; "
                "evaluating the general formula near them loses all precision"
            ),
        ),
    ],
    example=lambda: AdaptiveRobustLoss(alpha=1.0, c=1.0),
)
class AdaptiveRobustLoss(_RobustLoss):
    """Adaptive robust loss (ICLR 2024 Eq. (7)), with ``u = (x/c)^2``::

        f(x) = sum |alpha-2|/alpha * ((u/|alpha-2| + 1)^(alpha/2) - 1)

    ``alpha = 2`` gives ``u/2``, ``alpha = 0`` gives ``log(u/2 + 1)`` and
    ``alpha = -inf`` gives ``1 - exp(-u/2)``.
    """

    _equation = "7"

    def __init__(self, alpha: float, c: float = 1.0):
        super().__init__()
        _check_positive("c", c)
        if math.isnan(alpha) or alpha == math.inf:
            raise ValueError(f"alpha must be a real number or -inf, got {alpha}")
        self.alpha = alpha
        self.c = c
        self._certify_from_theorem5(admissible=alpha <= 2.0)

    def _s(self, x):
        a, u = self.alpha, (x / self.c) ** 2
        if a == 2.0:
            return 0.5 * u
        if a == 0.0:
            return torch.log1p(0.5 * u)
        if a == -math.inf:
            return -torch.expm1(-0.5 * u)
        b = abs(a - 2.0)
        return b / a * (torch.pow(u / b + 1.0, a / 2.0) - 1.0)

    def grad(self, x):
        a, u = self.alpha, (x / self.c) ** 2
        scale = x / self.c ** 2
        if a == 2.0:
            return scale
        if a == 0.0:
            return scale / (0.5 * u + 1.0)
        if a == -math.inf:
            return scale * torch.exp(-0.5 * u)
        b = abs(a - 2.0)
        return scale * torch.pow(u / b + 1.0, a / 2.0 - 1.0)


@documented(example=lambda: SquaredLogMinusRationalLoss())
class SquaredLogMinusRationalLoss(_RobustLoss):
    """``f(x) = sum log(1 + x^2) - x^2 / (2x^2 + 2)`` (ICLR 2024 Eq. (8)).

    The log-minus-rational form of :class:`LogMinusRationalPenalty`, applied to x^2.
    """

    _equation = "8"

    def __init__(self):
        super().__init__()
        self._certify_from_theorem5()

    def _s(self, x):
        x2 = x * x
        return torch.log1p(x2) - x2 / (2.0 * x2 + 2.0)

    def grad(self, x):
        x2 = x * x
        return 2.0 * x / (1.0 + x2) - x / (1.0 + x2) ** 2
