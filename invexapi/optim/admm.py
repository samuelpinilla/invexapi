from __future__ import annotations

from typing import Callable, Optional

import torch

from ..metadata import DesignDecision, Invariant, Provenance, Reference, documented
from ..functions import Composed, Sum
from ..operators import Identity, LinearOperator
from .base import Solver

__all__ = ["LinearizedADMM"]

_ADMM_REFERENCE = Reference(
    authors="Pinilla, S. and Thiyagalingam, J.",
    title="Global Optimality for Non-linear Constrained Restoration Problems via Invexity",
    venue="International Conference on Learning Representations (ICLR)",
    year=2024,
    locator="ADMM algorithm for invex-regularized restoration",
)


@documented(
    provenance=[
        Provenance(
            reference=_ADMM_REFERENCE,
            description=(
                "TV-regularized denoising via ADMM with a linearized/relaxed "
                "x-update (fixed alpha=0.5 extrapolation, fixed rho, no adaptive "
                "scheme); z-update and dual update are standard exact ADMM"
            ),
        )
    ],
    decisions=[
        DesignDecision(
            choice="name the class LinearizedADMM, not ADMM",
            rejected="calling it ADMM (unqualified)",
            rationale=(
                "the x-update is a single relaxed/extrapolated gradient step, not "
                "an exact ADMM subproblem solve - Linearized ADMM is a real, "
                "published variant name; calling it plain ADMM would imply a "
                "stronger per-iteration guarantee than this implementation gives"
            ),
        ),
        DesignDecision(
            choice="warn off Sum(smooth, Composed(penalty, D)) - the objective actually minimized",
            rejected="warning on smooth and penalty separately",
            rationale=(
                "certified parts do not make a certified sum (invex + invex is not "
                "invex), and penalty alone ignores D; Composed(penalty, D) lives on "
                "x's domain, so the true objective is one object, as in FISTA"
            ),
        ),
    ],
    invariants=[
        Invariant(
            "the primal residual used for the stopping criterion and the dual "
            "update must be computed from the NEWLY updated z, not the z from "
            "before this iteration's prox step - using the old z is trivially "
            "zero whenever x0 already minimizes smooth alone (e.g. x0=y for "
            "0.5||x-y||^2), causing a spurious immediate 'convergence'."
        )
    ],
)
class LinearizedADMM(Solver):
    """Linearized ADMM for ``min_x smooth(x) + penalty(D@x)``.

    Named ``LinearizedADMM``, not ``ADMM``, because its x-update is a single
    relaxed/extrapolated gradient step, not an exact ADMM subproblem solve.

    ``smooth`` needs ``.grad(x)`` (and optionally ``.value(x)``); ``penalty`` needs
    ``.prox(z, step)`` (and optionally ``.value(z)``), operating on ``D@x``'s
    domain, which may differ in shape from ``x`` (e.g. total-variation's ``D``
    stacks a vertical and horizontal gradient field). ``D`` defaults to
    :class:`~invexapi.operators.Identity`. ``project``, if given, is
    applied to ``x`` after every update (e.g. ``lambda x: x.clamp(min=0)`` for
    natural-image pixel constraints) - defaulting to
    no projection, since forcing non-negativity isn't appropriate for a general
    solver.
    """

    def __init__(
        self,
        smooth,
        penalty,
        D: Optional[LinearOperator] = None,
        rho: float = 1.0,
        alpha: float = 0.5,
        project: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        super().__init__(max_iter=max_iter, tol=tol)
        self.smooth = smooth
        self.penalty = penalty
        self.D = D if D is not None else Identity()
        self.rho = rho
        self.alpha = alpha
        self.project = project

    def run(self, x0: torch.Tensor):
        self._warn_if_unproven(Sum(self.smooth, Composed(self.penalty, self.D)), "smooth+penalty(D@x)")

        x = x0.clone()
        x_relaxed = x0.clone()
        z = self.D.apply(x0)
        d = torch.zeros_like(z)
        history = []

        for _ in range(self.max_iter):
            x1 = (1.0 - self.alpha) * x_relaxed + self.alpha * x
            grad_coupling = self.rho * self.D.adjoint(self.D.apply(x1) - z + d)
            grad_smooth = self.smooth.grad(x1)
            x_new = x - (1.0 / (2.0 * self.rho)) * (grad_coupling + grad_smooth)
            if self.project is not None:
                x_new = self.project(x_new)

            x_relaxed = (1.0 - self.alpha) * x_relaxed + self.alpha * x_new

            Dx = self.D.apply(x_new)
            z = self.penalty.prox(Dx + d / self.rho, 1.0 / self.rho)
            primal_residual = Dx - z
            d = d + self.rho * primal_residual

            if hasattr(self.smooth, "value") and hasattr(self.penalty, "value"):
                history.append((self.smooth.value(x_new) + self.penalty.value(Dx)).item())

            x = x_new

            # Standard ADMM stopping criterion: the primal residual Dx - z (not
            # x's own movement, which can be zero for a degenerate iteration or
            # two before the z/dual feedback has caught up, e.g. when x0 already
            # minimizes smooth alone) must vanish.
            if primal_residual.norm() < self.tol * max(Dx.norm(), 1.0):
                break

        return x_relaxed, history
