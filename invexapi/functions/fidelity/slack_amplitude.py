from __future__ import annotations

from typing import Optional

import torch

from ...metadata import DesignDecision, Provenance, Reference, documented
from ...core import Loss

__all__ = ["SlackAmplitudeLeastSquares"]

_WAVEMAX_REFERENCE = Reference(
    authors="Pinilla, S. et al.",
    title="WaveMax: Radar Waveform Design via Convex Maximization",
    venue="IEEE Transactions on Signal Processing, 73",
    year=2025,
    locator="Algorithm 2, lines 7-8 (data-fidelity step on |Tr(B Z)| vs sqrt(A))",
)


@documented(
    provenance=[
        Provenance(
            description=(
                "amplitude least squares 0.5*|| |z| + s - b ||^2 with its real "
                "gradient sgn(z)*(|z| + s - b), the data-fidelity term of the "
                "WaveMax FASTA step"
            ),
            reference=_WAVEMAX_REFERENCE,
        )
    ],
    decisions=[
        DesignDecision(
            choice="no convex/invex certificate",
            rejected="certifying via WaveMax Lemma 1 (convexity of problem (14))",
            rationale=(
                "Lemma 1 is about the constraint set |Tr(B Z)| <= sqrt(A); this "
                "amplitude least-squares loss is nonconvex in z (e.g. at z=0 "
                "whenever b - s > 0), so citing Lemma 1 for it would be wrong"
            ),
        )
    ],
)
class SlackAmplitudeLeastSquares(Loss):
    """Slack-relaxed amplitude least squares ``f(z) = 0.5*|| |z| + s - b ||^2``.

    Fits the magnitudes ``|z|`` of complex measurements to the targets ``b - s``:
    ``b >= 0`` holds the measured magnitudes and ``slack`` (``s >= 0``, default 0)
    relaxes them, e.g. ``s = max(b - |z_prev|, 0)`` refreshed between solver runs
    so that measurements already under their bound stop pulling ``|z|`` upward.
    ``grad`` is the real gradient ``sgn(z) * (|z| + s - b)`` for complex ``z``
    (treating C as R^2).
    """

    def __init__(self, b: torch.Tensor, slack: Optional[torch.Tensor] = None):
        super().__init__()
        self.b = b
        self.slack = torch.zeros_like(b) if slack is None else slack

    def _residual(self, z: torch.Tensor) -> torch.Tensor:
        return z.abs() + self.slack - self.b

    def value(self, z: torch.Tensor) -> torch.Tensor:
        r = self._residual(z)
        return 0.5 * torch.sum(r * r)

    def grad(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sgn(z) * self._residual(z)
