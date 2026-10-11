from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

from ..metadata import DesignDecision, Invariant, documented

__all__ = ["Operator", "LinearOperator", "NonlinearOperator", "PROPERTIES"]

# Declared structural properties. For a nonlinear operator they refer to its
# Jacobian at every x (e.g. full_row_rank: J_F(x) has full row rank for all x),
# which is the form the invexity composition results need.
PROPERTIES = ("full_row_rank", "injective", "unitary", "diagonal")

Shape = Optional[Tuple[int, ...]]


@documented(
    decisions=[
        DesignDecision(
            choice="Operator declares only apply; LinearOperator and NonlinearOperator refine it",
            rejected="a single LinearOperator base for every forward model",
            rationale=(
                "linearity is what licenses an adjoint (used by FASTA/LinearizedADMM); "
                "nonlinear forward models (e.g. a signal's ambiguity function) have "
                "no adjoint, only a Jacobian, which solvers obtain by autograd"
            ),
        ),
        DesignDecision(
            choice="structural properties are Optional[bool] attributes, None = unknown",
            rejected="computing properties numerically (rank via SVD) on construction",
            rationale=(
                "a numerical rank is evidence, not proof, and is unaffordable for "
                "matrix-free operators at image scale; operators set what is known "
                "exactly from their structure and users may declare() the rest"
            ),
        ),
    ],
    invariants=[
        Invariant(
            "None means unknown, never false: algebra only propagates a property when "
            "it provably follows from the operands', and otherwise yields None."
        ),
        Invariant(
            "in_shape/out_shape describe one unbatched input/output; None means the "
            "operator is shape-agnostic or the shape was not given."
        ),
    ],
)
class Operator(ABC):
    """A forward map ``x -> F(x)`` between tensors.

    Operators compose with ``@`` (``A @ B`` is ``x -> A(B(x))``; ``A @ x`` applies
    ``A`` to a tensor), add with ``+``/``-`` and scale with ``c * A``.
    """

    in_shape: Shape = None
    out_shape: Shape = None
    full_row_rank: Optional[bool] = None
    injective: Optional[bool] = None
    unitary: Optional[bool] = None
    diagonal: Optional[bool] = None

    @abstractmethod
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Return F(x)."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply(x)

    def declare(self, **properties: Optional[bool]) -> "Operator":
        """Assert structural properties known by the caller, e.g. ``declare(full_row_rank=True)``.

        Returns ``self`` for chaining. Declarations are taken on trust.
        """
        for name, value in properties.items():
            if name not in PROPERTIES:
                raise ValueError(f"unknown property {name!r}; expected one of {PROPERTIES}")
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be True, False or None, got {value!r}")
            setattr(self, name, value)
        return self

    def properties(self) -> dict:
        """The declared structural properties, as ``{name: Optional[bool]}``."""
        return {name: getattr(self, name) for name in PROPERTIES}

    def __matmul__(self, other):
        if isinstance(other, torch.Tensor):
            return self.apply(other)
        if isinstance(other, Operator):
            from .algebra import compose

            return compose(self, other)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Operator):
            from .algebra import add

            return add(self, other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Operator):
            from .algebra import add, scale

            return add(self, scale(-1.0, other))
        return NotImplemented

    def __neg__(self):
        from .algebra import scale

        return scale(-1.0, self)

    def __rmul__(self, c):
        if isinstance(c, (int, float, complex)):
            from .algebra import scale

            return scale(c, self)
        return NotImplemented

    __mul__ = __rmul__


class NonlinearOperator(Operator):
    """A differentiable nonlinear map ``x -> F(x)``.

    ``apply`` must be written in differentiable torch ops: solvers such as
    :class:`~invexapi.optim.LevenbergMarquardt` take its Jacobian with
    ``torch.func.jacfwd`` (treating complex tensors as pairs of reals).
    """


class LinearOperator(Operator):
    """A linear map ``x -> D@x`` plus its adjoint ``v -> D^T@v``.

    Used by :class:`~invexapi.optim.LinearizedADMM` to split a penalty applied to a
    transformed variable (e.g. total-variation regularization penalizes an image's
    gradient, not the image itself) from the smooth data-fidelity term, which is
    applied to the original variable, and by :class:`~invexapi.optim.FASTA` as the
    ``A`` in ``min_x f(A@x) + g(x)``. For complex operators, ``adjoint`` is the
    conjugate transpose (adjoint w.r.t. ``Re<u, v>``).
    """

    @abstractmethod
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Return D@x."""

    @abstractmethod
    def adjoint(self, v: torch.Tensor) -> torch.Tensor:
        """Return D^T@v, same shape as whatever `apply` was given."""

    @property
    def H(self) -> "LinearOperator":
        """The adjoint as an operator in its own right."""
        from .algebra import Adjoint

        return Adjoint(self)

    def norm(
        self,
        like: Optional[torch.Tensor] = None,
        max_iter: int = 200,
        tol: float = 1e-7,
        seed: int = 0,
    ) -> float:
        """Spectral norm ``||D||_2`` by power iteration on ``D^H D``.

        ``like`` fixes the input's shape/dtype/device; it defaults to a float64
        tensor of shape ``in_shape``. Gives the Lipschitz constant ``L = ||A||^2``
        of ``grad 0.5*||A x - y||^2``, so a safe FISTA step is ``1 / A.norm()**2``.
        """
        if like is None:
            if self.in_shape is None:
                raise ValueError("norm() needs `like` when the operator has no in_shape")
            like = torch.empty(self.in_shape, dtype=torch.float64)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(like.shape, generator=gen, dtype=like.real.dtype).to(like.device, like.dtype)
        x = x / torch.linalg.vector_norm(x)
        sigma = 0.0
        for _ in range(max_iter):
            y = self.adjoint(self.apply(x))
            y_norm = torch.linalg.vector_norm(y).item()
            if y_norm == 0.0:
                return 0.0
            new_sigma = y_norm ** 0.5
            x = y / y_norm
            if abs(new_sigma - sigma) <= tol * new_sigma:
                return new_sigma
            sigma = new_sigma
        return sigma
