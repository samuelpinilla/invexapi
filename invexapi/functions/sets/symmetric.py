from __future__ import annotations

import torch

from ...metadata import DesignDecision, Invariant, Provenance, documented
from ...core import Certificate, Penalty, Reference

__all__ = ["SymmetricNuclearPenalty"]

_CONVEXITY_REFERENCE = Reference(
    authors="Boyd, S. and Vandenberghe, L.",
    title="Convex Optimization",
    venue="Cambridge University Press",
    year=2004,
    locator="Section 3.1.5 (norms are convex); Section 2.2.1 (subspaces are convex)",
)

_SVT_REFERENCE = Reference(
    authors="Cai, J.-F., Candes, E. J. and Shen, Z.",
    title="A singular value thresholding algorithm for matrix completion",
    venue="SIAM Journal on Optimization, 20(4), 1956-1982",
    year=2010,
    locator="Theorem 2.1 (prox of the nuclear norm is singular value soft-thresholding)",
)


def _symmetric_part(X: torch.Tensor) -> torch.Tensor:
    return 0.5 * (X + X.transpose(-1, -2))


@documented(
    provenance=[
        Provenance(
            description="prox of lamb*||Z||_*: singular value soft-thresholding max(s - lamb*step, 0)",
            reference=_SVT_REFERENCE,
        )
    ],
    decisions=[
        DesignDecision(
            choice="prox = singular value soft-thresholding of the symmetric part (X + X^T)/2",
            rejected="soft-thresholding the raw input and symmetrizing afterwards",
            rationale=(
                "on the subspace of complex symmetric matrices ||Z - V||^2 = ||Z - sym(V)||^2 "
                "+ ||skew(V)||^2, so the constrained prox is the nuclear-norm prox of sym(V); "
                "that prox commutes with transposition and is unique, hence already symmetric "
                "(it soft-thresholds the Takagi values). The output is re-symmetrized only to "
                "remove rounding"
            ),
        ),
        DesignDecision(
            choice="symmetric (X^T = X), not Hermitian (X^H = X)",
            rejected="PSDTracePenalty's Hermitian PSD cone",
            rationale=(
                "lifting x -> x x^T (no conjugate) gives a complex symmetric rank-one matrix "
                "that is neither Hermitian nor PSD; the nuclear norm plays the role of the "
                "trace on the PSD cone"
            ),
        ),
    ],
    invariants=[
        Invariant("prox(V, t) is complex symmetric for every square V."),
        Invariant("prox(x x^T, t) == max(1 - lamb*t/||x||^2, 0) * x x^T for a rank-one symmetric input."),
    ],
    example=lambda: SymmetricNuclearPenalty(lamb=0.1),
)
class SymmetricNuclearPenalty(Penalty):
    """``g(Z) = lamb * ||Z||_* + indicator{Z = Z^T}`` on complex square matrices.

    The convex low-rank prior for complex symmetric lifts ``X = x x^T``. Its prox
    is singular-value soft-thresholding of the symmetric part; ``lamb = 0`` gives
    the plain projection onto symmetric matrices.
    """

    def __init__(self, lamb: float = 0.0, atol: float = 1e-8):
        super().__init__()
        if lamb < 0.0:
            raise ValueError(f"lamb must be non-negative, got {lamb}")
        self.lamb = lamb
        self.atol = atol

        convexity = Certificate(status="assumed", source="manual", reference=_CONVEXITY_REFERENCE)
        self._certify("convex", convexity)
        self._certify("invex", convexity)
        self._certify("quasi_convex", convexity)

    def value(self, Z: torch.Tensor) -> torch.Tensor:
        S = _symmetric_part(Z)
        nuclear = self.lamb * torch.linalg.svdvals(S).sum()
        tol = self.atol * max(Z.abs().max().item(), 1.0)
        if (Z - S).abs().max() > tol:
            return torch.full_like(nuclear, float("inf"))
        return nuclear

    def grad(self, Z: torch.Tensor) -> torch.Tensor:
        """A subgradient ``lamb * U_r V_r^H`` over the nonzero singular values of sym(Z)."""
        U, s, Vh = torch.linalg.svd(_symmetric_part(Z))
        keep = s > self.atol * max(s.max().item(), 1.0)
        return self.lamb * U[:, keep] @ Vh[keep]

    def prox(self, Z: torch.Tensor, step: float) -> torch.Tensor:
        U, s, Vh = torch.linalg.svd(_symmetric_part(Z))
        s = torch.clamp(s - self.lamb * step, min=0.0)
        return _symmetric_part((U * s.to(U.dtype)) @ Vh)

