from __future__ import annotations

import torch

from ...metadata import DesignDecision, Provenance, documented
from ...core import Certificate, Penalty, Reference

__all__ = ["PSDTracePenalty"]

_CONVEXITY_REFERENCE = Reference(
    authors="Boyd, S. and Vandenberghe, L.",
    title="Convex Optimization",
    venue="Cambridge University Press",
    year=2004,
    locator="Section 2.2.5 (PSD cone is convex); Section 3.1.5 (linear functions are convex)",
)

_WAVEMAX_REFERENCE = Reference(
    authors="Pinilla, S. et al.",
    title="WaveMax: Radar Waveform Design via Convex Maximization",
    venue="IEEE Transactions on Signal Processing, 73",
    year=2025,
    locator="Algorithm 2, lines 10-13",
)


def _hermitian_part(X: torch.Tensor) -> torch.Tensor:
    return 0.5 * (X + X.conj().transpose(-1, -2))


@documented(
    provenance=[
        Provenance(
            description=(
                "prox of lamb*tr(Z) + indicator_PSD(Z): eigenvalue shift-and-clip "
                "max(d - lamb*step, 0), the PSD-cone step of the WaveMax algorithm"
            ),
            reference=_WAVEMAX_REFERENCE,
        )
    ],
    decisions=[
        DesignDecision(
            choice="project the Hermitian part (X + X^H)/2 with eigh",
            rejected="a general (non-Hermitian) eigendecomposition of the raw input",
            rationale=(
                "the Frobenius projection of any square matrix onto the Hermitian "
                "PSD cone is the projection of its Hermitian part; eig() on a "
                "non-Hermitian matrix returns a non-unitary V, so V D V^H is not a "
                "projection at all"
            ),
        )
    ],
    example=lambda: PSDTracePenalty(lamb=0.1),
)
class PSDTracePenalty(Penalty):
    """``g(Z) = lamb * tr(Z) + indicator{Z Hermitian PSD}``.

    The prox is eigenvalue soft-thresholding onto the PSD cone: the trace term
    shifts every eigenvalue down by ``lamb*step`` before clipping at zero, which
    acts as a nuclear-norm (low-rank) prior on PSD matrices. ``lamb = 0`` gives the
    plain PSD-cone projection.
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
        H = _hermitian_part(Z)
        tol = self.atol * max(Z.abs().max().item(), 1.0)
        trace = self.lamb * torch.diagonal(H, dim1=-2, dim2=-1).sum().real
        if (Z - H).abs().max() > tol or torch.linalg.eigvalsh(H).min() < -tol:
            return torch.full_like(trace, float("inf"))
        return trace

    def grad(self, Z: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(Z.shape[-1], dtype=Z.dtype, device=Z.device)
        return self.lamb * eye

    def prox(self, Z: torch.Tensor, step: float) -> torch.Tensor:
        d, V = torch.linalg.eigh(_hermitian_part(Z))
        d = torch.clamp(d - self.lamb * step, min=0.0)
        return (V * d.to(V.dtype)) @ V.conj().transpose(-1, -2)
