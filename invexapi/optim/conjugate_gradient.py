from __future__ import annotations

from typing import Optional

import torch

from ..metadata import DesignDecision, Invariant, documented
from ._linesearch import backtracking
from .base import Solver

__all__ = ["NonlinearCG"]


@documented(
    decisions=[
        DesignDecision(
            choice="Polak-Ribière+ (PR beta clamped to >= 0, i.e. automatic restart)",
            rejected="plain Polak-Ribière or Fletcher-Reeves without restart",
            rationale=(
                "clamping the PR coefficient to be non-negative automatically "
                "restarts the direction to steepest descent whenever the plain PR "
                "formula would go negative, which is known to improve robustness "
                "on non-quadratic (e.g. invex) objectives over unrestarted variants"
            ),
        )
    ],
    invariants=[
        Invariant("beta_pr is always clamped to >= 0 before use as the restart coefficient.")
    ],
)
class NonlinearCG(Solver):
    """Nonlinear conjugate gradient (Polak-Ribière+, with restart) on any
    objective exposing ``.grad(x)`` (and, for the default backtracking line
    search, ``.value(x)``).

    Suited to general (invex or otherwise) differentiable objectives, unlike
    linear CG which only solves quadratic ``Ax=b`` systems.
    """

    def __init__(
        self,
        objective,
        step: Optional[float] = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        super().__init__(max_iter=max_iter, tol=tol)
        if step is None and not hasattr(objective, "value"):
            raise ValueError("objective must implement value(x) when step is None")
        self.objective = objective
        self.step = step

    def run(self, x0: torch.Tensor):
        self._warn_if_unproven(self.objective, "objective")
        x = x0.clone()
        history = []

        grad = self.objective.grad(x)
        direction = -grad

        for _ in range(self.max_iter):
            self._record(history, self.objective, x)
            if grad.norm() < self.tol:
                break

            if self.step is not None:
                t = self.step
            else:
                t = backtracking(self.objective.value, x, direction, grad)
            x = x + t * direction

            grad_new = self.objective.grad(x)
            beta_pr = torch.sum(grad_new * (grad_new - grad)) / torch.sum(grad * grad).clamp_min(1e-12)
            beta = torch.clamp(beta_pr, min=0.0)  # Polak-Ribière+ restart

            direction = -grad_new + beta * direction
            grad = grad_new

        return x, history
