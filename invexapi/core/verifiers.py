"""Verifiers that *produce* certificates rather than only reading them back."""

from __future__ import annotations

from typing import Any, Optional

import torch

from ..certificate import Certificate
from ..metadata import DesignDecision, Invariant, Reference, documented

__all__ = ["NumericalVerifier"]

_DEFINITION_4 = Reference(
    authors="Pinilla, S. and Thiyagalingam, J.",
    title="Global Optimality for Non-linear Constrained Restoration Problems via Invexity",
    venue="International Conference on Learning Representations (ICLR)",
    year=2024,
    locator="Definition 4 (spot-checked numerically on a grid)",
)


@documented(
    decisions=[
        DesignDecision(
            choice="return status='evidence', never 'assumed' or 'verified'",
            rejected="treating a passing grid check as a proof",
            rationale=(
                "a finite grid cannot establish a 'for all w > 0' condition; every "
                "certificate derived from this one inherits 'evidence' (the weakest "
                "status), so the doubt propagates instead of being laundered away"
            ),
        ),
        DesignDecision(
            choice="grid on [w_min, w_max] = [1e-3, 1e3] by default, relative tolerance on s/w^2",
            rejected="a grid reaching w -> 0",
            rationale=(
                "s(w)/w^2 suffers catastrophic cancellation as w -> 0 (e.g. "
                "log(1+w) - w/(1+w)), producing spurious ~1e-10 increases"
            ),
        ),
    ],
    invariants=[Invariant("check() returns None for any property it cannot spot-check")],
)
class NumericalVerifier:
    """Spot-checks ICLR 2024 Def. 4 (``admissible``) for functions exposing ``profile(w)``.

    For a separable ``g(x) = sum_i s(|x_i|)`` whose ``profile`` is ``s``, checks on a
    log-spaced grid that s(0) = 0, s is strictly increasing, and s(w)/w^2 is
    non-increasing. Use it to bring a user-written penalty into the rule engine::

        cert = NumericalVerifier().check(my_penalty, "admissible")
        if cert is not None:
            my_penalty._certify("admissible", cert)   # now Sum/Composed rules can fire
    """

    def __init__(self, w_min: float = 1e-3, w_max: float = 1e3, points: int = 4001, rtol: float = 1e-9):
        self.w_min, self.w_max, self.points, self.rtol = w_min, w_max, points, rtol

    def check(self, obj: Any, property_name: str) -> Optional[Certificate]:
        if property_name != "admissible" or not callable(getattr(obj, "profile", None)):
            return None
        w = torch.logspace(
            torch.log10(torch.tensor(self.w_min)).item(),
            torch.log10(torch.tensor(self.w_max)).item(),
            self.points,
            dtype=torch.float64,
        )
        with torch.no_grad():
            s0 = obj.profile(torch.zeros(1, dtype=torch.float64)).abs().item()
            s = obj.profile(w)
        ratio = s / (w * w)
        increasing = bool((s[1:] > s[:-1]).all()) and bool((s > 0).all())
        ratio_steps = ratio[1:] - ratio[:-1]
        non_increasing = bool((ratio_steps <= self.rtol * ratio[:-1].abs()).all())
        if s0 == 0.0 and increasing and non_increasing:
            return Certificate(status="evidence", source="NumericalVerifier", reference=_DEFINITION_4)
        return None
