from __future__ import annotations

import math
from typing import Callable, Optional

import torch

from ..metadata import DesignDecision, Invariant, Provenance, Reference, documented
from ..functions import Composed, Sum
from ..operators import Identity, LinearOperator
from .base import Solver

__all__ = ["FASTA"]

_FASTA_REFERENCE = Reference(
    authors="Goldstein, T., Studer, C. and Baraniuk, R.",
    title="A Field Guide to Forward-Backward Splitting with a FASTA Implementation",
    venue="arXiv:1411.3406",
    year=2014,
    locator="Sections 4 (adaptive BB step sizes), 5 (non-monotone backtracking), 6 (stopping)",
)

_STOP_RULES = ("hybrid", "residual", "normalized", "ratio", "iterations")


def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.vdot(a.reshape(-1), b.reshape(-1)).real.item()


def _norm(a: torch.Tensor) -> float:
    return torch.linalg.vector_norm(a).item()


@documented(
    provenance=[
        Provenance(
            description=(
                "FASTA's adaptive (Barzilai-Borwein 'adaptive' combination of the "
                "steepest-descent and minimum-residual step sizes), accelerated "
                "(FISTA with restart) and plain modes, non-monotone backtracking "
                "over a window of past f values, the two-point Lipschitz step "
                "estimate, and the residual/normalized/ratio/hybrid stopping "
                "rules; returns the best iterate seen, not the last"
            ),
            reference=_FASTA_REFERENCE,
        )
    ],
    decisions=[
        DesignDecision(
            choice="warn off Sum(Composed(smooth, A), penalty) - the objective actually minimized",
            rejected="warning on smooth and penalty separately",
            rationale=(
                "smooth acts on A@x, but Composed(smooth, A) lives on x's domain, so "
                "the true objective is one object whose certificate comes from the "
                "rules (e.g. convex_affine + convex_sum), as in FISTA/LinearizedADMM"
            ),
        ),
        DesignDecision(
            choice="optional user stop_now(x, iteration, residual, normalized_residual) hook",
            rejected="only the built-in stopping rules",
            rationale="keyword arguments replace an options struct; the hook keeps custom stopping possible",
        ),
    ],
    invariants=[
        Invariant(
            "A.adjoint must be the exact adjoint of A.apply under Re<u, v>; the "
            "backtracking condition and BB step sizes both assume it."
        ),
        Invariant(
            "the returned iterate is the one minimizing the objective (if "
            "record_objective) or the residual (otherwise): the method is "
            "non-monotone, so the last iterate need not be the best."
        ),
    ],
)
class FASTA(Solver):
    """Fast Adaptive Shrinkage/Thresholding for ``min_x smooth(A@x) + penalty(x)``.

    ``smooth`` needs ``.value(z)``/``.grad(z)`` on ``A``'s output; ``penalty`` needs
    ``.prox(x, step)`` (and ``.value(x)`` if ``record_objective``). ``A`` is a
    :class:`~invexapi.operators.LinearOperator` (default identity). Works for real
    and complex tensors (inner products are ``Re<u, v>``).

    Modes: ``adaptive=True`` (default) uses Barzilai-Borwein step sizes;
    ``accelerate=True`` uses FISTA momentum with optional restart; both False is
    plain forward-backward splitting. ``tau=None`` estimates the step from a
    two-point Lipschitz estimate ``tau = 0.2 / L``.

    ``run(x0)`` returns ``(x_best, history)``; ``history`` holds the objective per
    iteration when ``record_objective`` (else is empty), and ``self.info`` holds
    residuals, normalized residuals, step sizes, backtrack count and iterations.
    """

    def __init__(
        self,
        smooth,
        penalty,
        A: Optional[LinearOperator] = None,
        tau: Optional[float] = None,
        max_iter: int = 1000,
        tol: float = 1e-3,
        adaptive: bool = True,
        accelerate: bool = False,
        restart: bool = True,
        backtrack: bool = True,
        stepsize_shrink: Optional[float] = None,
        window: int = 10,
        max_backtracks: int = 20,
        eps_r: float = 1e-8,
        eps_n: float = 1e-8,
        stop_rule: str = "hybrid",
        stop_now: Optional[Callable[[torch.Tensor, int, float, float], bool]] = None,
        record_objective: bool = False,
        verbose: int = 0,
        generator: Optional[torch.Generator] = None,
    ):
        super().__init__(max_iter=max_iter, tol=tol)
        if stop_rule not in _STOP_RULES:
            raise ValueError(f"stop_rule must be one of {_STOP_RULES}, got {stop_rule!r}")
        self.smooth = smooth
        self.penalty = penalty
        self.A = A if A is not None else Identity()
        self.tau = tau
        self.adaptive = adaptive
        self.accelerate = accelerate
        self.restart = restart
        self.backtrack = backtrack
        if stepsize_shrink is None:
            stepsize_shrink = 0.2 if adaptive and not accelerate else 0.5
        self.stepsize_shrink = stepsize_shrink
        self.window = window
        self.max_backtracks = max_backtracks
        self.eps_r = eps_r
        self.eps_n = eps_n
        self.stop_rule = stop_rule
        self.stop_now = stop_now
        self.record_objective = record_objective
        self.verbose = verbose
        self.generator = generator
        self.info: dict = {}

    @property
    def mode(self) -> str:
        if self.accelerate:
            return "accelerated(FISTA)+restart" if self.restart else "accelerated(FISTA)"
        return "adaptive" if self.adaptive else "plain"

    def _full_grad(self, d: torch.Tensor) -> torch.Tensor:
        return self.A.adjoint(self.smooth.grad(d))

    def _estimate_tau(self, x0: torch.Tensor) -> float:
        x1 = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, generator=self.generator)
        x2 = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, generator=self.generator)
        g1 = self._full_grad(self.A.apply(x1))
        g2 = self._full_grad(self.A.apply(x2))
        lipschitz = max(_norm(g1 - g2) / _norm(x2 - x1), 1e-6)
        return 2.0 / lipschitz / 10.0

    def _should_stop(self, x, i, resid, norm_resid, max_resid) -> bool:
        if self.stop_now is not None:
            return bool(self.stop_now(x, i, resid, norm_resid))
        if self.stop_rule == "residual":
            return resid < self.tol
        if self.stop_rule == "normalized":
            return norm_resid < self.tol
        if self.stop_rule == "ratio":
            return resid / (max_resid + self.eps_r) < self.tol
        if self.stop_rule == "iterations":
            return False
        return resid / (max_resid + self.eps_r) < self.tol or norm_resid < self.tol

    def _objective(self, d: torch.Tensor, x: torch.Tensor) -> float:
        return self.smooth.value(d).item() + self.penalty.value(x).item()

    def run(self, x0: torch.Tensor):
        self._warn_if_unproven(Sum(Composed(self.smooth, self.A), self.penalty), "smooth(A@x)+penalty")

        tau1 = self.tau if self.tau is not None else self._estimate_tau(x0)
        if tau1 <= 0:
            raise ValueError(f"invalid step size: {tau1}")
        initial_tau = tau1

        residuals, normalized, taus, history = [], [], [], []
        f_vals = []
        total_backtracks = 0

        x1 = x0.clone()
        d1 = self.A.apply(x1)
        f1 = self.smooth.value(d1).item()
        f_vals.append(f1)
        gradf1 = self._full_grad(d1)

        if self.accelerate:
            x_accel1, d_accel1, alpha1 = x1, d1, 1.0

        max_residual = -math.inf
        best_value = math.inf
        best = x1
        i = 0

        for i in range(1, self.max_iter + 1):
            x0_, gradf0, tau0 = x1, gradf1, tau1

            x1hat = x0_ - tau0 * gradf0
            x1 = self.penalty.prox(x1hat, tau0)
            dx = x1 - x0_
            d1 = self.A.apply(x1)
            f1 = self.smooth.value(d1).item()

            backtracks = 0
            if self.backtrack:
                lo = max(i - self.window, 1) - 1
                m_ref = max(f_vals[lo : max(i - 1, 1)])
                while (
                    f1 - 1e-12 > m_ref + _dot(dx, gradf0) + _norm(dx) ** 2 / (2 * tau0)
                    and backtracks < self.max_backtracks
                ):
                    tau0 *= self.stepsize_shrink
                    x1hat = x0_ - tau0 * gradf0
                    x1 = self.penalty.prox(x1hat, tau0)
                    d1 = self.A.apply(x1)
                    f1 = self.smooth.value(d1).item()
                    dx = x1 - x0_
                    backtracks += 1
                total_backtracks += backtracks

            taus.append(tau0)
            resid = _norm(dx) / tau0
            residuals.append(resid)
            max_residual = max(max_residual, resid)
            normalizer = max(_norm(gradf0), _norm(x1 - x1hat) / tau0) + self.eps_n
            normalized.append(resid / normalizer)
            if i - 1 < len(f_vals):
                f_vals[i - 1] = f1
            else:
                f_vals.append(f1)

            if self.record_objective:
                value = self._objective(d1, x1)
                history.append(value)
            else:
                value = resid
            if value < best_value:
                best, best_value = x1, value

            if self.verbose > 1:
                print(f"{i}: resid = {resid:.2e}, backtrack = {backtracks}, tau = {tau0:.2e}")

            if self._should_stop(x1, i, resid, normalized[-1], max_residual) or i >= self.max_iter:
                break

            if self.adaptive and not self.accelerate:
                gradf1 = self._full_grad(d1)
                dg = gradf1 + (x1hat - x0_) / tau0
                dot = _dot(dx, dg)
                tau_s = _norm(dx) ** 2 / dot if dot != 0 else math.inf
                tau_m = max(dot / _norm(dg) ** 2, 0.0) if _norm(dg) > 0 else 0.0
                tau1 = tau_m if 2 * tau_m > tau_s else tau_s - 0.5 * tau_m
                if tau1 <= 0 or math.isinf(tau1) or math.isnan(tau1):
                    tau1 = tau0 * 1.5

            if self.accelerate:
                x_accel0, d_accel0, alpha0 = x_accel1, d_accel1, alpha1
                x_accel1, d_accel1 = x1, d1
                if self.restart and _dot(x0_ - x1, x1 - x_accel0) > 0:
                    alpha0 = 1.0
                alpha1 = (1 + math.sqrt(1 + 4 * alpha0**2)) / 2
                x1 = x_accel1 + (alpha0 - 1) / alpha1 * (x_accel1 - x_accel0)
                d1 = d_accel1 + (alpha0 - 1) / alpha1 * (d_accel1 - d_accel0)
                gradf1 = self._full_grad(d1)
                f_vals[i - 1] = self.smooth.value(d1).item()
                tau1 = tau0

            if not self.adaptive and not self.accelerate:
                gradf1 = self._full_grad(d1)
                tau1 = tau0

        self.info = {
            "mode": self.mode,
            "iterations": i,
            "residuals": residuals,
            "normalized_residuals": normalized,
            "stepsizes": taus,
            "backtracks": total_backtracks,
            "initial_stepsize": initial_tau,
        }
        if self.verbose:
            print(f"FASTA ({self.mode}): done in {i} iterations")
        return best, history
