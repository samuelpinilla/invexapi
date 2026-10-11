from __future__ import annotations

import torch

from ...core import Loss
from ...metadata import DesignDecision, Invariant, documented

__all__ = ["Sum"]


@documented(
    decisions=[
        DesignDecision(
            choice="derive Sum's certificates only through cited rules (invexapi.rules)",
            rejected="never deriving (the pre-Phase-2 behaviour), or deriving invex from invex parts",
            rationale=(
                "invex + invex is not invex in general, so invexity of the parts proves "
                "nothing; but convex + convex is convex (Boyd 3.2.1) and admissible + "
                "admissible is admissible, hence invex (ICLR 2024 Thm 3) - sound rules "
                "with citations, so the sum is certified exactly when one applies"
            ),
        )
    ],
    invariants=[
        Invariant(
            "a certificate attached via sum_instance._certify(...) overrides any "
            "derived one; with neither, the property is None (unproven)."
        )
    ],
)
class Sum(Loss):
    """``smooth(x) + penalty(x)``, e.g. the combined objective FISTA minimizes.

    Certificates are derived lazily from the parts' by :mod:`invexapi.rules`
    (``convex_sum``, ``admissible_sum``); see :func:`invexapi.rules.explain`.
    """

    def __init__(self, smooth: Loss, penalty: Loss):
        super().__init__()
        self.smooth = smooth
        self.penalty = penalty

    @property
    def parts(self) -> tuple:
        return (self.smooth, self.penalty)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.smooth.value(x) + self.penalty.value(x)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return self.smooth.grad(x) + self.penalty.grad(x)
