"""Operator algebra: composition, sums, scaling and adjoints.

Results are linear exactly when every operand is. Structural properties propagate
only along implications that always hold (listed beside each rule); anything not
implied is ``None`` (unknown), never ``False``.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..metadata import Invariant, documented
from .base import PROPERTIES, LinearOperator, NonlinearOperator, Operator

__all__ = [
    "compose",
    "add",
    "scale",
    "Composition",
    "LinearComposition",
    "Sum",
    "LinearSum",
    "Scaled",
    "LinearScaled",
    "Adjoint",
]


def _both(a: Optional[bool], b: Optional[bool]) -> Optional[bool]:
    return True if (a is True and b is True) else None


def _check_shapes(expected, got, what: str) -> None:
    if expected is not None and got is not None and tuple(expected) != tuple(got):
        raise ValueError(f"shape mismatch in {what}: {tuple(expected)} vs {tuple(got)}")


class _Composition(Operator):
    def __init__(self, outer: Operator, inner: Operator):
        _check_shapes(outer.in_shape, inner.out_shape, "composition")
        self.outer, self.inner = outer, inner
        self.in_shape, self.out_shape = inner.in_shape, outer.out_shape
        # Surjective, injective, unitary and diagonal maps are each closed under composition.
        for name in PROPERTIES:
            setattr(self, name, _both(getattr(outer, name), getattr(inner, name)))

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return self.outer.apply(self.inner.apply(x))


class _Sum(Operator):
    def __init__(self, left: Operator, right: Operator):
        _check_shapes(left.in_shape, right.in_shape, "sum (inputs)")
        _check_shapes(left.out_shape, right.out_shape, "sum (outputs)")
        self.left, self.right = left, right
        self.in_shape = left.in_shape if left.in_shape is not None else right.in_shape
        self.out_shape = left.out_shape if left.out_shape is not None else right.out_shape
        # Only diagonality survives a sum (rank, injectivity, unitarity can all be lost).
        self.diagonal = _both(left.diagonal, right.diagonal)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return self.left.apply(x) + self.right.apply(x)


class _Scaled(Operator):
    def __init__(self, c, base: Operator):
        self.c, self.base = c, base
        self.in_shape, self.out_shape = base.in_shape, base.out_shape
        self.diagonal = base.diagonal
        if c != 0:  # a nonzero scalar is invertible: rank and injectivity are unchanged
            self.full_row_rank, self.injective = base.full_row_rank, base.injective
        if abs(c) == 1:
            self.unitary = base.unitary

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return self.c * self.base.apply(x)


@documented(
    invariants=[
        Invariant(
            "compose/add/scale return a LinearOperator (with a correct adjoint) iff "
            "every operand is a LinearOperator, else a NonlinearOperator."
        )
    ]
)
class Composition(_Composition, NonlinearOperator):
    """``x -> outer(inner(x))`` where at least one factor is nonlinear."""


class LinearComposition(_Composition, LinearOperator):
    """``x -> outer(inner(x))``, adjoint ``inner^H(outer^H(v))``."""

    def adjoint(self, v: torch.Tensor) -> torch.Tensor:
        return self.inner.adjoint(self.outer.adjoint(v))


class Sum(_Sum, NonlinearOperator):
    """``x -> left(x) + right(x)`` where at least one term is nonlinear."""


class LinearSum(_Sum, LinearOperator):
    def adjoint(self, v: torch.Tensor) -> torch.Tensor:
        return self.left.adjoint(v) + self.right.adjoint(v)


class Scaled(_Scaled, NonlinearOperator):
    """``x -> c * base(x)`` for a nonlinear ``base``."""


class LinearScaled(_Scaled, LinearOperator):
    def adjoint(self, v: torch.Tensor) -> torch.Tensor:
        c = self.c.conjugate() if isinstance(self.c, complex) else self.c
        return c * self.base.adjoint(v)


class Adjoint(LinearOperator):
    """``v -> base^H(v)``; its own adjoint is ``base``."""

    def __init__(self, base: LinearOperator):
        self.base = base
        self.in_shape, self.out_shape = base.out_shape, base.in_shape
        # rank(D^H) = rank(D): D surjective <=> D^H injective, and vice versa.
        self.full_row_rank, self.injective = base.injective, base.full_row_rank
        self.unitary, self.diagonal = base.unitary, base.diagonal

    def apply(self, v: torch.Tensor) -> torch.Tensor:
        return self.base.adjoint(v)

    def adjoint(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.apply(x)

    @property
    def H(self) -> LinearOperator:
        return self.base


def _linear(*ops: Operator) -> bool:
    return all(isinstance(op, LinearOperator) for op in ops)


def compose(outer: Operator, inner: Operator) -> Operator:
    """``outer @ inner``."""
    return LinearComposition(outer, inner) if _linear(outer, inner) else Composition(outer, inner)


def add(left: Operator, right: Operator) -> Operator:
    """``left + right``."""
    return LinearSum(left, right) if _linear(left, right) else Sum(left, right)


def scale(c, base: Operator) -> Operator:
    """``c * base``."""
    return LinearScaled(c, base) if _linear(base) else Scaled(c, base)
