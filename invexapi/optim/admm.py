from __future__ import annotations

from typing import Callable, Optional

import torch

from ..metadata import DesignDecision, Invariant, Provenance, documented
from ..penalties.operators import Identity, LinearOperator
from .base import Solver

__all__ = ["LinearizedADMM"]


@documented(
    provenance=[
        Provenance(
            files=(
                "code from conference paper Global Optimality for Nonlinear "
                "Constrained Restoration Problems via Invexity",
            ),
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
            choice="warn on smooth and penalty separately, not via a combined Sum",
            rejected="reusing invexapi.penalties.Sum like FISTA does",
            rationale=(
                "Sum assumes smooth and penalty share x's domain; here penalty "
                "operates on D@x, generally a different shape (e.g. TV's D stacks "
                "a vertical+horizontal gradient field), so no single combined "
                "object can be constructed to certify"
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
    relaxed/extrapolated gradient step (transcribed exactly from the source ADMM
    scripts), not an exact ADMM subproblem solve.

    ``smooth`` needs ``.grad(x)`` (and optionally ``.value(x)``); ``penalty`` needs
    ``.prox(z, step)`` (and optionally ``.value(z)``), operating on ``D@x``'s
    domain, which may differ in shape from ``x`` (e.g. total-variation's ``D``
    stacks a vertical and horizontal gradient field). ``D`` defaults to
    :class:`~invexapi.penalties.operators.Identity`. ``project``, if given, is
    applied to ``x`` after every update (e.g. ``lambda x: x.clamp(min=0)`` for
    natural-image pixel constraints, matching the source scripts) - defaulting to
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
        # Sum (used by FISTA) assumes smooth and penalty share x's domain; here
        # penalty operates on D@x, generally a different shape, so there is no
        # single combined object to check - warn on each half separately instead.
        self._warn_if_unproven(self.smooth, "smooth")
        self._warn_if_unproven(self.penalty, "penalty")

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
