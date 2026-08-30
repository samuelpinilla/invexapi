from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch

from ..metadata import DesignDecision, Provenance, documented

__all__ = ["LinearOperator", "Identity", "FiniteDifference2D"]


class LinearOperator(ABC):
    """A linear map ``x -> D@x`` plus its adjoint ``v -> D^T@v``.

    Used by :class:`~invexapi.optim.LinearizedADMM` to split a penalty applied to a
    transformed variable (e.g. total-variation regularization penalizes an image's
    gradient, not the image itself) from the smooth data-fidelity term, which is
    applied to the original variable.
    """

    @abstractmethod
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Return D@x."""

    @abstractmethod
    def adjoint(self, v: torch.Tensor) -> torch.Tensor:
        """Return D^T@v, same shape as whatever `apply` was given."""


class Identity(LinearOperator):
    """D = I. The default operator for solvers that don't split on a transform."""

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def adjoint(self, v: torch.Tensor) -> torch.Tensor:
        return v


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
            files=("code_ICLR/ADMM/ADMM_Lq.py::tv_mtrx_2",),
            description=(
                "same forward-difference + 1/sqrt(12) scaling + zeroed-last-row/col "
                "boundary convention as tv_mtrx_2, reimplemented without matrix "
                "materialization"
            ),
        )
    ],
    decisions=[
        DesignDecision(
            choice="tensor-slicing implementation of the forward difference and its adjoint",
            rejected="materializing the sparse Kronecker-product matrix (tv_mtrx_2)",
            rationale=(
                "a materialized D matrix is memory-prohibitive at image sizes like "
                "2048x2048 (~4M pixels); the adjoint is verified against the "
                "matrix-free operator via a numerical inner-product identity test "
                "instead of comparing to an actual materialized matrix"
            ),
        )
    ],
)
class FiniteDifference2D(LinearOperator):
    """The 2D total-variation (forward-difference) operator and its adjoint.

    ``apply(x)`` returns a ``(2, H, W)`` stack of the vertical and horizontal
    forward differences of a ``(H, W)`` image, each scaled by ``1/sqrt(12)`` and
    with the last row/column forced to zero (free/Neumann boundary).
    """

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([_forward_diff(x, dim=-2), _forward_diff(x, dim=-1)], dim=0)

    def adjoint(self, v: torch.Tensor) -> torch.Tensor:
        dv, dh = v[0], v[1]
        return _adjoint_diff(dv, dim=-2) + _adjoint_diff(dh, dim=-1)
