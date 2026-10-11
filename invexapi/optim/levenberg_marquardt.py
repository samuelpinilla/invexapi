from __future__ import annotations

from typing import Optional

import torch
from torch.func import jacfwd

from ..metadata import DesignDecision, Invariant, Provenance, Reference, documented
from .base import Solver

__all__ = ["LevenbergMarquardt"]

_LM_REFERENCE = Reference(
    authors="Marquardt, D. W.",
    title="An Algorithm for Least-Squares Estimation of Nonlinear Parameters",
    venue="Journal of the Society for Industrial and Applied Mathematics, 11(2)",
    year=1963,
    locator="Sections 2-3 (damped Gauss-Newton step with diagonal scaling)",
)


def _as_real(t: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(t).reshape(-1) if t.is_complex() else t.reshape(-1)


@documented(
    provenance=[
        Provenance(
            description=(
                "Levenberg-Marquardt for min_x 0.5*||F(x) - y||^2: damped Gauss-Newton "
                "steps with Marquardt's column-norm scaling, damping decreased on "
                "success and increased on failure"
            ),
            reference=_LM_REFERENCE,
        )
    ],
    decisions=[
        DesignDecision(
            choice="solve each step as least squares on [J; sqrt(lam) D] with QR",
            rejected="the normal equations (J^T J + lam D^2) step = -J^T r",
            rationale=(
                "the normal equations square J's condition number; for problems "
                "with near-ambiguities (relative singular values ~1e-8) that "
                "squares to below machine precision and stalls the iteration"
            ),
        ),
        DesignDecision(
            choice="complex unknowns and residuals are handled as pairs of reals",
            rejected="Wirtinger (complex) Jacobians",
            rationale="forward-mode jacfwd on the real view is exact and needs no holomorphy",
        ),
    ],
    invariants=[
        Invariant(
            "operator.apply must be written in differentiable torch ops; the "
            "Jacobian is taken with torch.func.jacfwd, never finite differences."
        )
    ],
)
class LevenbergMarquardt(Solver):
    """Levenberg-Marquardt for nonlinear least squares ``min_x 0.5*||F(x) - y||^2``.

    ``operator`` is any :class:`~invexapi.operators.Operator` (typically a
    :class:`~invexapi.operators.NonlinearOperator`) with a differentiable
    ``apply``; ``x`` and ``F(x)`` may be real or complex. Each iteration forms the
    full Jacobian, so this suits problems with up to a few thousand real unknowns.
    Stops when the relative misfit ``||F(x) - y|| / ||y||`` drops below ``tol`` or
    no damped step decreases it.

    ``run(x0)`` returns ``(x, history)`` with the relative misfit after every
    accepted step; ``self.info`` records iterations and the final damping.
    """

    def __init__(
        self,
        operator,
        y: torch.Tensor,
        max_iter: int = 100,
        tol: float = 1e-15,
        damping: float = 1e-3,
        max_damping: float = 1e12,
    ):
        super().__init__(max_iter=max_iter, tol=tol)
        self.operator = operator
        self.y = y
        self.damping = damping
        self.max_damping = max_damping
        self.info: dict = {}

    def run(self, x0: torch.Tensor):
        self._warn_if_unproven(self.operator, "least-squares objective")
        complex_x = x0.is_complex()
        shape = x0.shape
        y = _as_real(self.y)
        y_norm = max(torch.linalg.vector_norm(y).item(), 1e-300)

        def unpack(v: torch.Tensor) -> torch.Tensor:
            return torch.view_as_complex(v.reshape(*shape, 2)) if complex_x else v.reshape(shape)

        def residual(v: torch.Tensor) -> torch.Tensor:
            return _as_real(self.operator.apply(unpack(v))) - y

        v = _as_real(x0.detach().clone()).clone()
        r = residual(v)
        cost = torch.linalg.vector_norm(r).item()
        lam = self.damping
        history = [cost / y_norm]
        i = 0
        for i in range(1, self.max_iter + 1):
            if history[-1] < self.tol:
                break
            J = jacfwd(residual)(v)
            scale = torch.linalg.vector_norm(J, dim=0).clamp(min=1e-300)
            rhs = torch.cat([-r, torch.zeros_like(v)])
            accepted = False
            while lam < self.max_damping:
                Q, R = torch.linalg.qr(torch.cat([J, lam**0.5 * torch.diag(scale)]))
                step = torch.linalg.solve_triangular(R, (Q.T @ rhs).unsqueeze(-1), upper=True)
                r_new = residual(v + step.squeeze(-1))
                cost_new = torch.linalg.vector_norm(r_new).item()
                if cost_new < cost:
                    v, r, cost = v + step.squeeze(-1), r_new, cost_new
                    lam = max(lam / 5.0, 1e-20)
                    accepted = True
                    break
                lam *= 4.0
            if not accepted:
                break
            history.append(cost / y_norm)

        self.info = {"iterations": i, "damping": lam, "misfit": history[-1]}
        return unpack(v), history
