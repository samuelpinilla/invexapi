from __future__ import annotations

from typing import Optional, Sequence

import torch

from ..metadata import DesignDecision, Invariant, Provenance, Reference, documented
from .base import LinearOperator, NonlinearOperator
from .frft import _OZAKTAS_REFERENCE, frft_matrix

__all__ = ["frft_orders", "FrFTAmbiguity", "LiftedFrFTAmbiguity"]

_WAVEMAX_REFERENCE = Reference(
    authors="Pinilla, S. et al.",
    title="WaveMax: Radar Waveform Design via Convex Maximization",
    venue="IEEE Transactions on Signal Processing, 73",
    year=2025,
    locator="Eqs. (9), (12), (13)",
)


def _frft_stack(
    n: int,
    orders: torch.Tensor,
    pad: int,
    dtype: torch.dtype,
    device: Optional[torch.device],
) -> torch.Tensor:
    """``(K, pad*n, n)``: FrFT matrices of size ``pad*n`` restricted to the ``n``
    columns where a length-``n`` signal sits when centred on the larger grid."""
    if pad < 1:
        raise ValueError(f"pad must be >= 1, got {pad}")
    m = pad * n
    lo = m // 2 - n // 2
    return torch.stack(
        [frft_matrix(m, float(a), dtype=dtype, device=device)[:, lo : lo + n] for a in orders]
    )


def frft_orders(n: int, decimation: int = 1, low: float = 0.5, high: float = 1.5) -> torch.Tensor:
    """``ceil(n/decimation)`` FrFT orders uniformly spaced in ``[low, high]``.

    Orders are in units of pi/2, so the default range is the angles
    ``[pi/4, 3pi/4]``; ``decimation`` keeps every ``decimation``-th order of a full
    ``n``-order sampling.
    """
    count = -(-n // decimation)
    return torch.linspace(low, high, count, dtype=torch.float64)


@documented(
    provenance=[
        Provenance(
            description=(
                "the lifted measurement Tr(B_{alpha,k} X) with "
                "B_{alpha,k} = F_alpha^H diag(DFT row k) F_alpha, whose magnitude "
                "squared at X = x x^H is the FrFT-based ambiguity function"
            ),
            reference=_WAVEMAX_REFERENCE,
        )
    ],
    decisions=[
        DesignDecision(
            choice="optional centred zero-padding of the signal (pad) before the FrFT",
            rejected="always transforming on the signal's own N-sample grid",
            rationale=(
                "the fast FrFT is accurate only for signals whose time-frequency "
                "support fits well inside its sqrt(N)-diameter normalization circle; "
                "an off-centre or chirped pulse leaks, which breaks the continuous "
                "model's exact time-shift/Doppler invariance of the ambiguity "
                "function at ~1e-7 relative level (pad=1). pad=2 restores it to "
                "~1e-10 and makes the shift a numerically exact ambiguity"
            ),
            reference=_OZAKTAS_REFERENCE,
        ),
        DesignDecision(
            choice="matrix-free: fft(diag(F_a X F_a^H)) from the K precomputed N x N FrFT matrices",
            rejected="materializing all N*K matrices B_{alpha,k}",
            rationale=(
                "N*K dense N x N complex matrices take ~1.4 GB at N=128, K=43 and "
                "cost one trace per measurement; the factored form needs K N x N "
                "matrices and one batched matmul"
            ),
        )
    ],
    invariants=[
        Invariant(
            "apply(x x^H) == fft(|F_alpha x|^2) for every alpha, so |apply(x x^H)|^2 "
            "is the FrFT-based ambiguity function A[alpha, k]."
        ),
        Invariant("adjoint is the exact conjugate transpose of apply under Re<U, V>."),
    ],
)
class LiftedFrFTAmbiguity(LinearOperator):
    """Lifted FrFT-based ambiguity-function measurement ``X -> Tr(B_{alpha,k} X)``.

    ``apply`` maps an ``(N, N)`` matrix ``X`` to a ``(K, M)`` tensor indexed
    ``[alpha, k]`` (one row per FrFT order); for ``X = x x^H`` its magnitude squared
    is the FrFT-based ambiguity function, i.e. the ambiguity function sampled
    along the rotated (FrFT-order) directions of the time-frequency plane.
    ``adjoint`` maps ``(K, M)`` back to ``(N, N)``. ``B_{alpha,k}`` is not
    Hermitian, so ``apply`` is complex even for Hermitian ``X``.

    With ``pad > 1`` the signal is centred in ``M = pad*N`` zeros before every
    FrFT, so measurements are ``(K, M)``; this makes the fast FrFT markedly more
    accurate (see the design decisions).
    """

    def __init__(
        self,
        n: int,
        orders: Sequence[float],
        pad: int = 1,
        dtype: torch.dtype = torch.complex128,
        device: Optional[torch.device] = None,
    ):
        self.n = n
        self.pad = pad
        self.m = pad * n
        self.orders = torch.as_tensor(orders, dtype=torch.float64)
        self.matrices = _frft_stack(n, self.orders, pad, dtype, device)
        self.in_shape = (n, n)
        self.out_shape = (len(self.orders), self.m)

    @property
    def shape(self):
        """Measurement shape ``(K, M)``, ``M = pad * N``."""
        return (len(self.orders), self.m)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """``(K, M)`` FrFTs ``F_alpha @ x`` of a signal ``x`` of length N."""
        return self.matrices @ x.to(self.matrices.dtype)

    def measure(self, x: torch.Tensor) -> torch.Tensor:
        """``apply(x x^H)`` computed from the signal directly, in O(K N^2)."""
        return torch.fft.fft(self.transform(x).abs() ** 2, dim=-1)

    def ambiguity(self, x: torch.Tensor) -> torch.Tensor:
        """The FrFT-based ambiguity function ``A[alpha, k] = |apply(x x^H)|^2``."""
        return self.measure(x).abs() ** 2

    def signal_operator(self) -> "FrFTAmbiguity":
        """The nonlinear map ``x -> A[alpha, k]`` on signals, sharing these matrices."""
        return FrFTAmbiguity(self.n, self.orders, pad=self.pad, matrices=self.matrices)

    def apply(self, X: torch.Tensor) -> torch.Tensor:
        F = self.matrices
        diag = ((F @ X.to(F.dtype)) * F.conj()).sum(dim=-1)
        return torch.fft.fft(diag, dim=-1)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        F = self.matrices
        c = self.m * torch.fft.ifft(y.to(F.dtype), dim=-1)
        return (F.conj().transpose(-1, -2) @ (c.unsqueeze(-1) * F)).sum(dim=0)


def _abs2(z: torch.Tensor) -> torch.Tensor:
    # |z|^2 as real^2 + imag^2: smooth everywhere, unlike abs(), so forward-mode
    # Jacobians stay exact at measurements that vanish
    return z.real**2 + z.imag**2


@documented(
    decisions=[
        DesignDecision(
            choice="fit intensities A[alpha, k] (a quartic polynomial in x), not magnitudes sqrt(A)",
            rejected="the magnitude map x -> |Tr(B (x x^H))|",
            rationale=(
                "|z| is not differentiable at z = 0, and many ambiguity-function "
                "samples are near zero; Gauss-Newton on the magnitude residual "
                "loses its quadratic rate there, while the intensity residual is "
                "a polynomial"
            ),
        )
    ],
)
class FrFTAmbiguity(NonlinearOperator):
    """The FrFT-based ambiguity function of a signal, ``x -> A[alpha, k]``.

    ``A[alpha, k] = |sum_n |(F_alpha x)[n]|^2 e^{-2 pi i k n / M}|^2``, a ``(K, M)``
    real tensor (``M = pad * N``); this is the signal-domain (nonlinear)
    counterpart of :class:`LiftedFrFTAmbiguity`, whose ``apply(x x^H)`` it squares
    in magnitude.
    """

    def __init__(
        self,
        n: int,
        orders: Sequence[float],
        pad: int = 1,
        dtype: torch.dtype = torch.complex128,
        device: Optional[torch.device] = None,
        matrices: Optional[torch.Tensor] = None,
    ):
        self.n = n
        self.pad = pad
        self.orders = torch.as_tensor(orders, dtype=torch.float64)
        if matrices is None:
            matrices = _frft_stack(n, self.orders, pad, dtype, device)
        self.matrices = matrices
        self.in_shape = (n,)
        self.out_shape = (len(self.orders), matrices.shape[-2])

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        intensity = _abs2(self.matrices @ x.to(self.matrices.dtype))
        return _abs2(torch.fft.fft(intensity.to(self.matrices.dtype), dim=-1))
