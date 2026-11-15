from __future__ import annotations

import torch

from ...metadata import DesignDecision, Invariant, Provenance, Reference, documented
from ...core import Certificate, Loss

__all__ = ["AmplitudeHingeSquared"]

_CONVEXITY_REFERENCE = Reference(
    authors="Bauschke, H. H. and Combettes, P. L.",
    title="Convex Analysis and Monotone Operator Theory in Hilbert Spaces",
    venue="Springer, 2nd ed.",
    year=2017,
    locator=(
        "Chapter 12 (0.5*d_C^2 is the Moreau envelope of the indicator of a closed "
        "convex set C: convex and Frechet differentiable with gradient Id - P_C)"
    ),
)

_PHASEMAX_REFERENCE = Reference(
    authors="Goldstein, T. and Studer, C.",
    title="PhaseMax: Convex Phase Retrieval via Basis Pursuit",
    venue="IEEE Transactions on Information Theory, 64(4)",
    year=2018,
    locator="quadratic penalty 0.5*||max(|Ax| - b, 0)||^2 on the constraints |Ax| <= b",
)

_WAVEMAX_REFERENCE = Reference(
    authors="Pinilla, S. et al.",
    title="WaveMax: Radar Waveform Design via Convex Maximization",
    venue="IEEE Transactions on Signal Processing, 73",
    year=2025,
    locator="problem (14), constraints |Tr(B Z)| <= sqrt(A); Lemma 1 (their convexity)",
)


@documented(
    provenance=[
        Provenance(
            description=(
                "squared hinge 0.5*||max(|z| - b, 0)||^2 with real gradient "
                "sgn(z)*max(|z| - b, 0): the quadratic penalty of the amplitude "
                "constraints |z| <= b"
            ),
            reference=_PHASEMAX_REFERENCE,
        ),
        Provenance(
            description=(
                "as the smooth term of the lifted WaveMax problem it penalizes "
                "violations of |Tr(B_{alpha,k} Z)| <= sqrt(A[alpha, k])"
            ),
            reference=_WAVEMAX_REFERENCE,
        ),
    ],
    decisions=[
        DesignDecision(
            choice="certify convex and C^1 as 0.5*dist^2 to the product of disks {|z_i| <= b_i}",
            rejected="certifying via WaveMax Lemma 1",
            rationale=(
                "Lemma 1 only says the constraint set is convex; the property used "
                "here is that half the squared distance to a closed convex set is "
                "convex and C^1, and max(|z_i| - b_i, 0) is exactly the distance from "
                "z_i to the disk of radius b_i >= 0"
            ),
        ),
        DesignDecision(
            choice="reject negative b",
            rejected="accepting any real b",
            rationale="the disk {|z| <= b} is empty for b < 0, so the distance reading (and the certificate) would not hold",
        ),
    ],
    invariants=[
        Invariant(
            "grad(z) == z - P_C(z) with P_C(z) = z * min(1, b/|z|): the gradient of "
            "0.5*d_C^2, zero exactly on the feasible set."
        )
    ],
    example=lambda: AmplitudeHingeSquared(torch.ones(1)),
)
class AmplitudeHingeSquared(Loss):
    """Squared amplitude hinge ``f(z) = 0.5*|| max(|z| - b, 0) ||^2``.

    The quadratic penalty of the bound constraints ``|z| <= b``: zero wherever the
    magnitudes of the (complex) measurements stay under ``b >= 0`` and quadratic
    in the violation otherwise. Unlike a two-sided amplitude fit it is convex,
    so ``Composed(f, A)`` is convex for any linear ``A`` (rule ``convex_affine``).
    ``grad`` is the real gradient ``sgn(z) * max(|z| - b, 0)`` (treating C as R^2).
    """

    def __init__(self, b: torch.Tensor):
        super().__init__()
        if (b < 0).any():
            raise ValueError("b must be non-negative")
        self.b = b

        convexity = Certificate(status="assumed", source="manual", reference=_CONVEXITY_REFERENCE)
        self._certify("convex", convexity)
        self._certify("differentiable", convexity)

    def _excess(self, z: torch.Tensor) -> torch.Tensor:
        return torch.clamp(z.abs() - self.b, min=0.0)

    def value(self, z: torch.Tensor) -> torch.Tensor:
        e = self._excess(z)
        return 0.5 * torch.sum(e * e)

    def grad(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sgn(z) * self._excess(z)
