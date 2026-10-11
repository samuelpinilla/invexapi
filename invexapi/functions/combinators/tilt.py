from __future__ import annotations

import torch

from ...metadata import DesignDecision, Invariant, documented
from ...core import Penalty

__all__ = ["LinearTilt"]


def _real_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.vdot(a.reshape(-1), b.reshape(-1)).real


@documented(
    decisions=[
        DesignDecision(
            choice="LinearTilt carries no certificate unless one is attached explicitly",
            rejected="inheriting the wrapped penalty's certificates",
            rationale=(
                "a linear tilt preserves convexity but NOT invexity in general "
                "(adding a linear term can create stationary points that are not "
                "global minima); no rule for tilts is registered yet, so nothing is "
                "derived - a convex-tilt rule (convex + linear = convex) would be "
                "sound and can be added to invexapi.rules when needed"
            ),
        )
    ],
    invariants=[
        Invariant(
            "prox(x, t) == base.prox(x + t*c, t): the tilt only shifts the prox "
            "argument, so any base penalty with a prox gains a tilted one for free."
        )
    ],
)
class LinearTilt(Penalty):
    """``g(x) = base(x) - Re<c, x>``: a penalty plus a linear objective term.

    E.g. PhaseMax/WaveMax maximize alignment ``Re<x0, x>`` with an anchor, which in
    minimization form is a tilt by ``c = x0``.
    """

    def __init__(self, base: Penalty, c: torch.Tensor):
        super().__init__()
        self.base = base
        self.c = c

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.value(x) - _real_inner(self.c, x)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.grad(x) - self.c

    def prox(self, x: torch.Tensor, step: float) -> torch.Tensor:
        return self.base.prox(x + step * self.c, step)
