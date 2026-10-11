"""Finite-difference (total-variation) operators."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

from ..metadata import DesignDecision, Provenance, Reference, documented
from .base import LinearOperator

__all__ = ["FiniteDifference1D", "FiniteDifference2D"]

_TV_REFERENCE = Reference(
    authors="Pinilla, S. and Thiyagalingam, J.",
    title="Global Optimality for Non-linear Constrained Restoration Problems via Invexity",
    venue="International Conference on Learning Representations (ICLR)",
    year=2024,
    locator="total-variation regularized restoration experiments",
)


def _forward_diff(x: torch.Tensor, dim: int) -> torch.Tensor:
    diff = torch.zeros_like(x)
    if dim == -2:
        diff[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    else:
        diff[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
    return diff / math.sqrt(12.0)


def _adjoint_diff(p: torch.Tensor, dim: int) -> torch.Tensor:
    out = torch.zeros_like(p)
    if dim == -2:
        out[..., 0, :] = -p[..., 0, :]
        out[..., 1:-1, :] = p[..., :-2, :] - p[..., 1:-1, :]
        out[..., -1, :] = p[..., -2, :]
    else:
        out[..., :, 0] = -p[..., :, 0]
        out[..., :, 1:-1] = p[..., :, :-2] - p[..., :, 1:-1]
        out[..., :, -1] = p[..., :, -2]
    return out / math.sqrt(12.0)


@documented(
    provenance=[
        Provenance(
            description=(
                "anisotropic total-variation operator with the forward-difference, "
                "1/sqrt(12) scaling and zeroed-last-row/column boundary convention "
                "used in the paper's TV experiments"
            ),
            reference=_TV_REFERENCE,
        )
    ],
    decisions=[
        DesignDecision(
            choice="tensor-slicing implementation of the forward difference and its adjoint",
            rejected="materializing D as a sparse Kronecker-product matrix",
            rationale=(
                "a materialized D matrix is memory-prohibitive at image sizes like "
                "2048x2048 (~4M pixels); the adjoint is verified against the "
                "matrix-free operator via a numerical inner-product identity test "
                "instead of comparing to an actual materialized matrix"
            ),
        ),
        DesignDecision(
            choice="declare full_row_rank=False and injective=False",
            rejected="leaving both unknown",
            rationale=(
                "both are structural facts: the output (2HW entries, with a zeroed "
                "last row/column) outnumbers the input, and constant images lie in "
                "the kernel; invexity of g(Dx) for TV comes from g being admissible "
                "(ICLR 2024 Thm 4), not from D's rank"
            ),
        ),
    ],
)
class FiniteDifference2D(LinearOperator):
    """The 2D total-variation (forward-difference) operator and its adjoint.

    ``apply(x)`` returns a ``(2, H, W)`` stack of the vertical and horizontal
    forward differences of a ``(H, W)`` image, each scaled by ``1/sqrt(12)`` and
    with the last row/column forced to zero (free/Neumann boundary). ``shape``
    (``(H, W)``) is optional and only fills in ``in_shape``/``out_shape``.
    """

    full_row_rank = False
    injective = False
    unitary = False
    diagonal = False

    def __init__(self, shape: Optional[Sequence[int]] = None):
        if shape is not None:
            self.in_shape = tuple(shape)
            self.out_shape = (2,) + self.in_shape

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([_forward_diff(x, dim=-2), _forward_diff(x, dim=-1)], dim=0)

    def adjoint(self, v: torch.Tensor) -> torch.Tensor:
        dv, dh = v[0], v[1]
        return _adjoint_diff(dv, dim=-2) + _adjoint_diff(dh, dim=-1)


@documented(
    provenance=[
        Provenance(
            description="the (n-1) x n first-order difference matrix of the 1-D invex TV result",
            reference=Reference(
                authors=_TV_REFERENCE.authors,
                title=_TV_REFERENCE.title,
                venue=_TV_REFERENCE.venue,
                year=_TV_REFERENCE.year,
                locator="Theorem 4 and Eq. (23)",
            ),
        )
    ],
    decisions=[
        DesignDecision(
            choice="output length n-1 (no zero padding), declared full_row_rank=True",
            rejected="FiniteDifference2D's zero-padded, 1/sqrt(12)-scaled convention",
            rationale=(
                "Theorem 4's invexity of g(Dx) rests on D having full row rank, which "
                "a zero-padded output row would break; rows (-1, 1) at distinct "
                "offsets are linearly independent for every n"
            ),
        )
    ],
)
class FiniteDifference1D(LinearOperator):
    """``x -> x[1:] - x[:-1]`` along the last dim (``R^n -> R^(n-1)``), and its adjoint."""

    full_row_rank = True
    injective = False
    unitary = False
    diagonal = False

    def __init__(self, n: Optional[int] = None):
        if n is not None:
            if n < 2:
                raise ValueError(f"n must be at least 2, got {n}")
            self.in_shape, self.out_shape = (n,), (n - 1,)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., 1:] - x[..., :-1]

    def adjoint(self, v: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros_like(v[..., :1])
        return torch.cat([zero, v], dim=-1) - torch.cat([v, zero], dim=-1)
