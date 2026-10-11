from __future__ import annotations

import torch

from ...core import Penalty
from ...metadata import DesignDecision, Invariant, Provenance, Reference, documented
from ._admissible import ADMISSIBLE_DEFINITION, admissible_certificate

__all__ = ["LLqPenalty"]

_LLQ_REFERENCE = Reference(
    **ADMISSIBLE_DEFINITION,
    locator="Theorem 4 (log-quasinorm regularizer) and the TV restoration experiments",
)

_BISECTION_STEPS = 100


def _gp(z, q, beta):
    """g'(z) for g(z) = log(1 + beta z^q) / beta, z > 0."""
    return q * z ** (q - 1.0) / (1.0 + beta * z ** q)


def _gpp(z, q, beta):
    """g''(z) = q z^(q-2) ((q-1) - beta z^q) / (1 + beta z^q)^2  (< 0, increasing)."""
    zq = z ** q
    return q * z ** (q - 2.0) * ((q - 1.0) - beta * zq) / (1.0 + beta * zq) ** 2


def _g(z, q, beta):
    return torch.log1p(beta * z ** q) / beta


def _prox_magnitude(u: torch.Tensor, lam: torch.Tensor, q: float, beta: float) -> torch.Tensor:
    """Global minimizer over z >= 0 of h(z) = lam*g(z) + 0.5*(z - u)^2, elementwise."""
    tiny = torch.finfo(u.dtype).tiny
    hi = torch.clamp(u, min=tiny)

    # 1) h'' = 1 + lam*g'' is increasing, so h' has a single minimum at z_m where
    #    h'' crosses 0; bisect in log space over (hi*1e-20, hi].
    lo_m = hi * 1e-20
    hi_m = hi.clone()
    convex_everywhere = 1.0 + lam * _gpp(lo_m, q, beta) >= 0
    concave_everywhere = 1.0 + lam * _gpp(hi, q, beta) <= 0
    for _ in range(_BISECTION_STEPS):
        mid = torch.sqrt(lo_m * hi_m)
        neg = 1.0 + lam * _gpp(mid, q, beta) < 0
        lo_m = torch.where(neg, mid, lo_m)
        hi_m = torch.where(neg, hi_m, mid)
    z_m = torch.where(convex_everywhere, lo_m, torch.where(concave_everywhere, hi, hi_m))

    # 2) a positive local minimizer exists iff h'(z_m) < 0; it is the root of h' in
    #    (z_m, u] (h'(u) = lam*g'(u) > 0 there).
    hp_m = lam * _gp(z_m, q, beta) + z_m - u
    has_root = hp_m < 0
    a, b = z_m.clone(), hi.clone()
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (a + b)
        neg = lam * _gp(mid, q, beta) + mid - u < 0
        a = torch.where(neg, mid, a)
        b = torch.where(neg, b, mid)
    root = 0.5 * (a + b)

    # 3) keep it only if it beats z = 0 (ties go to the root, as for the log penalties).
    h_root = lam * _g(root, q, beta) + 0.5 * (root - u) ** 2
    keep = has_root & (h_root <= 0.5 * u * u)
    return torch.where(keep, root, torch.zeros_like(root))


@documented(
    decisions=[
        DesignDecision(
            choice="exact implicit-function backward at the output y",
            rejected="the reference kernel's gradient 1 - lam*g''(u), evaluated at the input",
            rationale=(
                "the kernel's formula is only a first-order approximation of "
                "dy/du = 1/(1 + lam*g''(y)); the exact form is verified by gradcheck "
                "and is what unrolled networks training through the prox need"
            ),
        )
    ],
)
class _LLqProx(torch.autograd.Function):
    """Bisection forward (no autograd unrolling) + exact implicit backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: torch.Tensor, q: float, beta: float) -> torch.Tensor:
        with torch.no_grad():
            sign = torch.where(x < 0, -torch.ones_like(x), torch.ones_like(x))
            z = _prox_magnitude(x.abs(), lam.to(x.dtype), q, beta)
        ctx.save_for_backward(z, sign, lam)
        ctx.q, ctx.beta = q, beta
        return sign * z

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        z, sign, lam = ctx.saved_tensors
        q, beta = ctx.q, ctx.beta
        active = z > 0
        safe = torch.where(active, z, torch.ones_like(z))
        denom = 1.0 + lam * _gpp(safe, q, beta)
        dz_du = torch.where(active, 1.0 / denom, torch.zeros_like(z))
        grad_x = dz_du * grad_y
        grad_lam = None
        if ctx.needs_input_grad[1]:
            dz_dlam = torch.where(active, -_gp(safe, q, beta) / denom, torch.zeros_like(z))
            grad_lam = (sign * dz_dlam * grad_y).sum().reshape(lam.shape)
        return grad_x, grad_lam, None, None


@documented(
    provenance=[
        Provenance(
            description=(
                "log-quasinorm penalty (lamb/beta)*log(1 + beta|x|^q) and its prox by "
                "bisection on the stationarity condition, as in the paper's TV "
                "restoration experiments"
            ),
            reference=_LLQ_REFERENCE,
        )
    ],
    decisions=[
        DesignDecision(
            choice="the (lamb/beta)*log(1 + beta|x|^q) form, certified admissible by Def. 4 inspection",
            rejected="Eq. (3)'s log(1 + (|x| + eps)^p), which Theorem 4 names directly",
            rationale=(
                "the paper's experiments solve this form; it satisfies Def. 4 exactly "
                "(s(0) = 0), whereas Eq. (3) has s(0) = log(1 + eps^p) != 0"
            ),
        ),
        DesignDecision(
            choice="return the global minimizer: locate h's inflection, bisect h' beyond it, compare with z=0",
            rejected="the reference kernel's bracket [(lamb q)^(1/(2-q)), |x|] without a comparison",
            rationale=(
                "that bracket can return a local minimizer that loses to z=0 just above "
                "the threshold; h'' is increasing (|g''| strictly decreasing for q <= 1), "
                "so the two-stage bisection finds the only candidate"
            ),
        ),
    ],
    invariants=[
        Invariant(
            "tests/reference_numpy.py::llq_prox_np transcribes the paper's kernel; it "
            "must agree with prox() away from the threshold, where the kernel is exact"
        )
    ],
    example=lambda: LLqPenalty(lamb=0.1, q=0.5),
)
class LLqPenalty(Penalty):
    """Log-quasinorm penalty ``g(x) = (lamb/beta) * sum log(1 + beta*|x|^q)``, ``0 < q <= 1``."""

    def __init__(self, lamb: float, q: float, beta: float = 1.0):
        super().__init__()
        if lamb <= 0.0:
            raise ValueError(f"lamb must be positive, got {lamb}")
        if not 0.0 < q <= 1.0:
            raise ValueError(f"q must be in (0, 1], got {q}")
        if beta <= 0.0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.lamb, self.q, self.beta = lamb, q, beta
        self._certify("admissible", admissible_certificate("(lamb/beta) * log(1 + beta*w^q), 0 < q <= 1"))

    def profile(self, w: torch.Tensor) -> torch.Tensor:
        """Scalar profile s with g(x) = sum s(|x_i|)."""
        return self.lamb * _g(w, self.q, self.beta)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.profile(x.abs()).sum()

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return self.lamb * torch.sign(x) * _gp(x.abs(), self.q, self.beta)

    def prox(self, x: torch.Tensor, step: float) -> torch.Tensor:
        lam = torch.as_tensor(self.lamb * step, dtype=x.dtype, device=x.device)
        return _LLqProx.apply(x, lam, self.q, self.beta)
