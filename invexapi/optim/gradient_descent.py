from __future__ import annotations

from typing import Optional

import torch

from ._linesearch import backtracking
from .base import Solver

__all__ = ["GradientDescent"]


class GradientDescent(Solver):
    """Gradient descent on any objective exposing ``.grad(x)`` (and, for the
    default backtracking line search, ``.value(x)``).

    If ``step`` is given, a fixed step size is used; otherwise each iteration
    picks its step via Armijo backtracking.
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
        for _ in range(self.max_iter):
            grad = self.objective.grad(x)
            self._record(history, self.objective, x)
            if grad.norm() < self.tol:
                break
            if self.step is not None:
                t = self.step
            else:
                t = backtracking(self.objective.value, x, -grad, grad)
            x = x - t * grad
        return x, history
