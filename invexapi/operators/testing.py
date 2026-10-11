"""Numerical checks for user-written operators."""

from __future__ import annotations

from typing import Optional

import torch

from .base import LinearOperator

__all__ = ["check_adjoint"]


def _real_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.vdot(a.reshape(-1), b.reshape(-1)).real


def check_adjoint(
    op: LinearOperator,
    x: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    tol: float = 1e-8,
    seed: int = 0,
) -> float:
    """Check ``Re<A x, v> == Re<x, A^H v>`` on random inputs; return the relative error.

    ``x`` defaults to a float64 Gaussian of shape ``op.in_shape``; ``v`` to a
    Gaussian shaped (and typed) like ``op(x)``. Raises ``AssertionError`` if the
    relative error exceeds ``tol`` - loosen ``tol`` for float32 operators.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if x is None:
        if op.in_shape is None:
            raise ValueError("check_adjoint() needs `x` when the operator has no in_shape")
        x = torch.randn(op.in_shape, generator=gen, dtype=torch.float64)
    ax = op.apply(x)
    if v is None:
        v = torch.randn(ax.shape, generator=gen, dtype=ax.real.dtype)
        if ax.is_complex():
            v = torch.complex(v, torch.randn(ax.shape, generator=gen, dtype=ax.real.dtype))
        v = v.to(ax.device)
    lhs = _real_inner(ax, v)
    rhs = _real_inner(x.to(ax.dtype) if ax.is_complex() else x, op.adjoint(v))
    scale = torch.linalg.vector_norm(ax) * torch.linalg.vector_norm(v)
    err = (abs(lhs - rhs) / scale.clamp_min(torch.finfo(scale.dtype).tiny)).item()
    if err > tol:
        raise AssertionError(f"adjoint mismatch: relative error {err:.3e} > tol {tol:.1e}")
    return err
