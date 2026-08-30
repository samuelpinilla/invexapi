from __future__ import annotations

import torch


def backtracking(
    value_fn,
    x: torch.Tensor,
    direction: torch.Tensor,
    grad: torch.Tensor,
    step0: float = 1.0,
    shrink: float = 0.5,
    c1: float = 1e-4,
    max_backtracks: int = 50,
) -> float:
    """Armijo backtracking line search along ``direction`` from ``x``.

    Returns a step size ``t`` such that ``value_fn(x + t*direction)`` gives
    sufficient decrease, or the smallest tried step if none satisfies it.
    """
    f0 = value_fn(x)
    slope = torch.sum(grad * direction)
    t = step0
    for _ in range(max_backtracks):
        if value_fn(x + t * direction) <= f0 + c1 * t * slope:
            return t
        t *= shrink
    return t
