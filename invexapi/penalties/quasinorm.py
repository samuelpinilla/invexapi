from __future__ import annotations

import torch

from ..metadata import DesignDecision, Invariant, Provenance, documented
from .base import Certificate, Penalty, Reference

__all__ = ["QuasinormInvexPenalty"]

_KERNEL_PROVENANCE = Provenance(
    files=(
        "codes from conference paper Improved Imaging by Invex Regularizers" 
        "with Global Optima Guarantees",
    ),
    description="byte-identical to implementation in Improved Imaging by Invex Regularizers" 
                "with Global Optima Guarantees",
)

_INVEXITY_REFERENCE = Reference(
    authors="Pinilla, Mu, Bourne, Thiyagalingam",
    title="Improved Imaging by Invex Regularizers with Global Optima Guarantees",
    venue="NeurIPS",
    year=2022,
    locator="Lemma 1, item 1 (Eq. 6)",
)


def _tol(lamb: torch.Tensor, q: float) -> torch.Tensor:
    return (lamb * q * (1.0 - q)) ** (1.0 / (2.0 - q))


def _beta(lamb: torch.Tensor, q: float) -> torch.Tensor:
    return (2.0 * lamb * (1.0 - q)) ** (1.0 / (2.0 - q))


@documented(
    provenance=[_KERNEL_PROVENANCE],
    decisions=[
        DesignDecision(
            choice="custom analytic torch.autograd.Function backward",
            rejected="plain autograd through the 4-step Newton fixed-point loop in forward",
            rationale=(
                "the paper's analytic gradient (gradG kernel) avoids autograd having "
                "to save every iteration's activations to backprop through the loop"
            ),
        )
    ],
    invariants=[
        Invariant(
            "any change to the forward fixed-point solve must have a matching "
            "change to backward's analytic gradient formula, or the two will "
            "silently diverge (there is no autograd check tying them together)."
        )
    ],
)
class _QuasinormProx(torch.autograd.Function):
    """Analytic forward+backward for the quasinorm invex prox."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lamb: torch.Tensor, q: float) -> torch.Tensor:
        ctx.save_for_backward(x, lamb)
        ctx.q = q

        absx = x.abs()
        sign = torch.sign(x)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)

        tol = _tol(lamb, q)
        beta = _beta(lamb, q)
        tau = beta + lamb * q * beta ** (q - 1.0)

        z = beta + (absx - beta) / 2.0
        for _ in range(4):
            z = absx - lamb * q * z ** (q - 1.0)

        y = torch.where(absx == tau, sign * beta, sign * z)
        y = torch.where((absx < tau) | (absx < beta), torch.zeros_like(y), y)
        y = torch.where(absx >= tol, y, torch.zeros_like(y))
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x, lamb = ctx.saved_tensors
        q = ctx.q

        absx = x.abs()
        sign = torch.sign(x)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        tol = _tol(lamb, q)
        active = absx >= tol

        grad_lamb_elem = torch.where(
            active, -q * absx ** (q - 1.0) * sign * grad_y, torch.zeros_like(grad_y)
        )
        grad_x = torch.where(
            active,
            (1.0 - lamb * q * (q - 1.0) * absx ** (q - 2.0)) * grad_y,
            torch.zeros_like(grad_y),
        )
        grad_lamb = grad_lamb_elem.sum().reshape_as(lamb) if lamb.requires_grad else None
        return grad_x, grad_lamb, None


@documented(
    provenance=[_KERNEL_PROVENANCE],
    example=lambda: QuasinormInvexPenalty(lamb=0.1, q=0.5),
)
class QuasinormInvexPenalty(Penalty):
    """Invex penalty ``g(x) = lamb * |x|^q``, ``0 < q < 1``.

    Its proximal operator is a 4-step Newton fixed-point solve; 
    ``value``/``grad`` are the closed-form penalty and its
    derivative, used by gradient-based solvers.
    """

    def __init__(self, lamb: float, q: float):
        super().__init__()
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q}")
        if lamb <= 0.0:
            raise ValueError(f"lamb must be positive, got {lamb}")
        self.lamb = lamb
        self.q = q
        self._certify("invex", Certificate(status="assumed", source="manual", reference=_INVEXITY_REFERENCE))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.lamb * x.abs().pow(self.q).sum()

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return self.lamb * self.q * x.abs().pow(self.q - 1.0) * torch.sign(x)

    def prox(self, x: torch.Tensor, step: float) -> torch.Tensor:
        lamb = torch.as_tensor(self.lamb * step, dtype=x.dtype, device=x.device)
        return _QuasinormProx.apply(x, lamb, self.q)
