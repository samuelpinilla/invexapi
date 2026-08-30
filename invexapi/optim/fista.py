from __future__ import annotations

import math

import torch

from ..metadata import DesignDecision, documented
from ..penalties import Sum
from .base import Solver

__all__ = ["FISTA"]


@documented(
    decisions=[
        DesignDecision(
            choice="warn/record off self.combined = Sum(smooth, penalty), not smooth/penalty individually",
            rejected="checking smooth's and penalty's certificates separately",
            rationale=(
                "a global-optimum claim is about the COMBINED objective smooth+penalty "
                "minimizes, and certificates don't compose from the parts (see "
                "invexapi.penalties.Sum) - checking the parts separately could miss "
                "that the combination itself is uncertified even when both halves are"
            ),
        )
    ],
)
class FISTA(Solver):
    """Accelerated proximal gradient (FISTA) for ``min_x smooth(x) + penalty(x)``.

    ``smooth`` must implement ``.grad(x)`` (and optionally ``.value(x)`` for the
    tracked history); ``penalty`` must implement ``.prox(x, step)`` (and optionally
    ``.value(x)``). ``step`` should be <= 1/L where L is the Lipschitz constant of
    ``smooth.grad``.
    """

    def __init__(self, smooth, penalty, step: float, max_iter: int = 100, tol: float = 1e-6):
        super().__init__(max_iter=max_iter, tol=tol)
        self.smooth = smooth
        self.penalty = penalty
        self.combined = Sum(smooth, penalty)
        self.step = step

    def run(self, x0: torch.Tensor):
        self._warn_if_unproven(self.combined, "smooth+penalty")
        x_prev = x0.clone()
        y = x0.clone()
        t_prev = 1.0
        history = []

        for _ in range(self.max_iter):
            grad = self.smooth.grad(y)
            x = self.penalty.prox(y - self.step * grad, self.step)

            self._record(history, self.combined, x)

            if (x - x_prev).norm() < self.tol * max(x_prev.norm(), 1.0):
                x_prev = x
                break

            t = (1.0 + math.sqrt(1.0 + 4.0 * t_prev ** 2)) / 2.0
            y = x + ((t_prev - 1.0) / t) * (x - x_prev)

            x_prev = x
            t_prev = t

        return x_prev, history
