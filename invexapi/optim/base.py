from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch

from ..metadata import DesignDecision, documented
from ._certification import warn_if_unproven

__all__ = ["Solver"]


@documented(
    decisions=[
        DesignDecision(
            choice="standardize only max_iter/tol and the run(x0)->(x,history) contract",
            rejected="a fully unified constructor signature across all solvers",
            rationale=(
                "GradientDescent/NonlinearCG take a single duck-typed objective "
                "while FISTA takes smooth+penalty — a real shape difference, not "
                "incidental duplication to force into one signature"
            ),
        )
    ],
)
class Solver(ABC):
    """Common shape for GradientDescent, FISTA, and NonlinearCG."""

    def __init__(self, max_iter: int = 100, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol

    @abstractmethod
    def run(self, x0: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        """Run the solver from ``x0``, returning ``(x_final, objective_history)``."""

    def _warn_if_unproven(self, obj, label: str) -> None:
        warn_if_unproven(obj, label)

    def _record(self, history: List[float], obj, x: torch.Tensor) -> None:
        """Append ``obj.value(x)`` to ``history`` if ``obj`` exposes ``value``."""
        if hasattr(obj, "value"):
            history.append(obj.value(x).item())
