"""Generic linear operators: identity, dense matrices, diagonals, convolutions, masks."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from ..metadata import DesignDecision, Invariant, documented
from .base import LinearOperator

__all__ = ["Identity", "MatrixOperator", "Diagonal", "Convolution", "Subsample"]


class Identity(LinearOperator):
    """D = I. The default operator for solvers that don't split on a transform."""

    full_row_rank = True
    injective = True
    unitary = True
    diagonal = True

    def __init__(self, shape: Optional[Sequence[int]] = None):
        if shape is not None:
            self.in_shape = self.out_shape = tuple(shape)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def adjoint(self, v: torch.Tensor) -> torch.Tensor:
        return v


class MatrixOperator(LinearOperator):
    """``x -> M @ x`` for a dense ``(m, n)`` matrix, applied along the last dim.

    Structural properties are unknown (``None``) unless declared, e.g.
    ``MatrixOperator(M).declare(full_row_rank=True)`` - a float matrix's rank is a
    numerical judgement, not something this class asserts on its own.
    """

    def __init__(self, matrix: torch.Tensor):
        if matrix.ndim != 2:
            raise ValueError(f"matrix must be 2-D, got shape {tuple(matrix.shape)}")
        self.matrix = matrix
        self.out_shape, self.in_shape = (matrix.shape[0],), (matrix.shape[1],)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.matrix.dtype) @ self.matrix.T

    def adjoint(self, v: torch.Tensor) -> torch.Tensor:
        return v.to(self.matrix.dtype) @ self.matrix.conj()


class Diagonal(LinearOperator):
    """``x -> d * x`` (entrywise), adjoint ``v -> conj(d) * v``.

    Rank and injectivity are exact here: both hold iff no entry of ``d`` is zero.
    """

    diagonal = True

    def __init__(self, d: torch.Tensor):
        self.d = d
        self.in_shape = self.out_shape = tuple(d.shape)
        nonzero = bool((d != 0).all())
        self.full_row_rank = self.injective = nonzero

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return self.d * x

    def adjoint(self, v: torch.Tensor) -> torch.Tensor:
        return self.d.conj() * v


@documented(
    decisions=[
        DesignDecision(
            choice="circular (periodic-boundary) convolution computed by FFT",
            rejected="zero-padded 'same' convolution via torch.nn.functional.conv2d",
            rationale=(
                "circular convolution is diagonalized by the DFT, so the adjoint is "
                "an exact conjugate multiply and the spectrum gives the operator's "
                "singular values for free; boundary handling can be layered on with "
                "a Subsample (crop) composed after it"
            ),
        )
    ],
    invariants=[
        Invariant(
            "the kernel's centre (index k//2 along each axis) is the origin, so a "
            "centred delta kernel is the identity"
        )
    ],
)
class Convolution(LinearOperator):
    """Circular convolution of an ``shape``-sized signal with ``kernel`` (any ndim).

    Applies over the trailing ``kernel.ndim`` dims; leading dims are batch dims.
    """

    def __init__(self, kernel: torch.Tensor, shape: Sequence[int]):
        shape = tuple(shape)
        if len(shape) != kernel.ndim or any(k > s for k, s in zip(kernel.shape, shape)):
            raise ValueError(f"kernel {tuple(kernel.shape)} does not fit signal shape {shape}")
        self.kernel = kernel
        self.in_shape = self.out_shape = shape
        self._dims = tuple(range(-kernel.ndim, 0))

        padded = torch.zeros(shape, dtype=kernel.dtype, device=kernel.device)
        padded[tuple(slice(0, k) for k in kernel.shape)] = kernel
        shifts = tuple(-(k // 2) for k in kernel.shape)
        self.transfer = torch.fft.fftn(torch.roll(padded, shifts, dims=self._dims), dim=self._dims)

    def _filter(self, x: torch.Tensor, transfer: torch.Tensor) -> torch.Tensor:
        out = torch.fft.ifftn(torch.fft.fftn(x, dim=self._dims) * transfer, dim=self._dims)
        return out if (x.is_complex() or self.kernel.is_complex()) else out.real

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return self._filter(x, self.transfer)

    def adjoint(self, v: torch.Tensor) -> torch.Tensor:
        return self._filter(v, self.transfer.conj())


class Subsample(LinearOperator):
    """Keep the entries where boolean ``mask`` is True: ``x -> x[mask]`` (a 1-D vector).

    The adjoint scatters back into zeros. Always full row rank (its rows are
    distinct rows of the identity); injective iff the mask keeps every entry.
    """

    full_row_rank = True

    def __init__(self, mask: torch.Tensor):
        if mask.dtype != torch.bool:
            raise TypeError("mask must be a boolean tensor")
        self.mask = mask
        self.in_shape = tuple(mask.shape)
        self.out_shape = (int(mask.sum()),)
        self.injective = bool(mask.all())

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.mask]

    def adjoint(self, v: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(v.shape[:-1] + self.in_shape, dtype=v.dtype, device=v.device)
        out[..., self.mask] = v
        return out
