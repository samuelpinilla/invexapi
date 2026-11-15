from __future__ import annotations

from typing import Optional, Sequence

import torch

from ..metadata import DesignDecision, Invariant, Provenance, Reference, documented
from .base import LinearOperator, NonlinearOperator

__all__ = ["RadarAmbiguity", "LiftedRadarAmbiguity", "integer_grid"]

_RADAR_PR_REFERENCE = Reference(
    authors="Pinilla, S., Mishra, K. V., Sadler, B. M. & Arguello, H.",
    title="Phase retrieval for radar waveform design",
    venue="Information and Inference: A Journal of the IMA, 13(3), iaae015",
    year=2024,
    locator="Eqs. (2), (4)",
)


def integer_grid(n: int, decimation: int = 1) -> torch.Tensor:
    """``0, L, 2L, ... < n`` as float64: every ``decimation``-th integer delay or Doppler bin."""
    if decimation < 1:
        raise ValueError(f"decimation must be >= 1, got {decimation}")
    return torch.arange(0, n, decimation, dtype=torch.float64)


def _grids(
    n: int,
    delays: Optional[Sequence[float] | torch.Tensor],
    dopplers: Optional[Sequence[float] | torch.Tensor],
    dtype: torch.dtype,
    device: Optional[torch.device],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Delay/Doppler grids plus the factors ``D[tau, l] = e^{-2 pi i l tau / n}``
    (centred frequencies ``l in [-n/2, n/2)``) and ``E[f, m] = e^{-2 pi i m f / n}``."""
    delay_grid = integer_grid(n) if delays is None else torch.as_tensor(delays, dtype=torch.float64)
    dopplers_grid = integer_grid(n) if dopplers is None else torch.as_tensor(dopplers, dtype=torch.float64)
    freqs = torch.fft.fftfreq(n, d=1.0 / n, dtype=torch.float64)
    time = torch.arange(n, dtype=torch.float64)
    D = torch.exp(-2j * torch.pi * torch.outer(delay_grid, freqs) / n).to(dtype=dtype, device=device)
    E = torch.exp(-2j * torch.pi * torch.outer(dopplers_grid, time) / n).to(dtype=dtype, device=device)
    return delay_grid, dopplers_grid, D, E


@documented(
    provenance=[
        Provenance(
            description=(
                "the lifted measurement Tr(B_{tau,f} X) with "
                "B_{tau,f} = Phi_tau^H diag(e^{-2 pi i n f / N}), whose magnitude squared "
                "at X = x x^H is the discrete (cyclic) narrow-band ambiguity function"
            ),
            reference=_RADAR_PR_REFERENCE,
        )
    ],
    decisions=[
        DesignDecision(
            choice=(
                "delays as a Fourier-domain phase ramp Phi_tau = F^H diag(e^{-2 pi i l tau/N}) F "
                "(centred l) and Doppler as a time-domain ramp, both on arbitrary real grids"
            ),
            rejected="integer cyclic index shifts x[n - p] and DFT Doppler bins k only",
            rationale=(
                "the ramps reproduce Eq. (2) exactly on integer grids, extend it to "
                "fractional delays/Doppler, and make incomplete ambiguity functions "
                "(removed delays or Doppler bins) just a choice of grid"
            ),
        ),
        DesignDecision(
            choice="matrix-free: ifft of X along rows, one (N, N) x (N, T) and one (K, N) x (N, T) matmul",
            rejected="materializing the T*K matrices B_{tau,f}",
            rationale=(
                "T*K dense N x N matrices are 268M complex entries at N=128; the "
                "factored form costs O(N^2 (T + K)) per application"
            ),
        ),
    ],
    invariants=[
        Invariant(
            "apply(x x^H)[tau, f] == sum_n x[n] conj(x[n - tau]) e^{-2 pi i n f / N} for integer "
            "tau, f, so |apply(x x^H)|^2 is the ambiguity function A[p, k] of Eq. (2)."
        ),
        Invariant("adjoint is the exact conjugate transpose of apply under Re<U, V>."),
        Invariant("on the full integer grid (T = K = N), adjoint(apply(X)) == N * X."),
    ],
)
class LiftedRadarAmbiguity(LinearOperator):
    """Lifted narrow-band ambiguity-function measurement ``X -> Tr(B_{tau,f} X)``.

    ``apply`` maps an ``(N, N)`` matrix ``X`` to a ``(T, K)`` tensor indexed
    ``[tau, f]`` (one row per delay, one column per Doppler); for ``X = x x^H``
    its magnitude squared is the ambiguity function
    ``A[tau, f] = |sum_n x[n] conj((Phi_tau x)[n]) e^{-2 pi i n f / N}|^2``,
    where ``Phi_tau`` delays by ``tau`` samples through a Fourier-domain phase
    ramp. ``delays`` and ``dopplers`` default to the integers ``0..N-1``, which
    is Eq. (2) of Pinilla et al. (2024) verbatim. ``B_{tau,f}`` is not
    Hermitian, so ``apply`` is complex even for Hermitian ``X``.
    """

    def __init__(
        self,
        n: int,
        delays: Optional[Sequence[float] | torch.Tensor] = None,
        dopplers: Optional[Sequence[float] | torch.Tensor] = None,
        dtype: torch.dtype = torch.complex128,
        device: Optional[torch.device] = None,
    ):
        self.n = n
        self.delays, self.dopplers, self.D, self.E = _grids(n, delays, dopplers, dtype, device)
        # P[m, l] = e^{-2 pi i l m / n}: undoes the row ifft at the diagonal m
        self.P = torch.fft.fft(torch.eye(n, dtype=dtype, device=device), dim=0)
        self.in_shape = (n, n)
        self.out_shape = (len(self.delays), len(self.dopplers))

    @property
    def shape(self):
        """Measurement shape ``(T, K)``."""
        return self.out_shape

    def delay(self, x: torch.Tensor) -> torch.Tensor:
        """``(T, N)`` delayed copies ``Phi_tau x`` of a signal ``x``."""
        return torch.fft.ifft(torch.fft.fft(x.to(self.D.dtype)) * self.D, dim=-1)

    def measure(self, x: torch.Tensor) -> torch.Tensor:
        """``apply(x x^H)`` computed from the signal directly, in O(T N (log N + K))."""
        g = x.to(self.D.dtype) * self.delay(x).conj()
        return g @ self.E.T

    def ambiguity(self, x: torch.Tensor) -> torch.Tensor:
        """The ambiguity function ``A[tau, f] = |apply(x x^H)|^2``."""
        return self.measure(x).abs() ** 2

    def signal_operator(self) -> "RadarAmbiguity":
        """The nonlinear map ``x -> A[tau, f]`` on signals, sharing these grids."""
        return RadarAmbiguity(self.n, self.delays, self.dopplers, factors=(self.D, self.E))

    def apply(self, X: torch.Tensor) -> torch.Tensor:
        # (X Phi_tau^H)[m, m] = sum_l ifft(X)[m, l] P[m, l] conj(D[tau, l])
        M = torch.fft.ifft(X.to(self.D.dtype), dim=-1) * self.P
        G = M @ self.D.conj().T
        return (self.E @ G).T

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        G = self.E.conj().T @ y.to(self.D.dtype).T
        M = (G @ self.D) * self.P.conj()
        return torch.fft.fft(M, dim=-1) / self.n


def _abs2(z: torch.Tensor) -> torch.Tensor:
    # |z|^2 as real^2 + imag^2: smooth everywhere, unlike abs(), so forward-mode
    # Jacobians stay exact at measurements that vanish
    return z.real**2 + z.imag**2


@documented(
    decisions=[
        DesignDecision(
            choice="fit intensities A[tau, f] (a quartic polynomial in x), not magnitudes sqrt(A)",
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
class RadarAmbiguity(NonlinearOperator):
    """The narrow-band ambiguity function of a signal, ``x -> A[tau, f]``.

    ``A[tau, f] = |sum_n x[n] conj((Phi_tau x)[n]) e^{-2 pi i n f / N}|^2``, a
    ``(T, K)`` real tensor; the signal-domain (nonlinear) counterpart of
    :class:`LiftedRadarAmbiguity`, whose ``apply(x x^H)`` it squares in magnitude.
    """

    def __init__(
        self,
        n: int,
        delays: Optional[Sequence[float] | torch.Tensor] = None,
        dopplers: Optional[Sequence[float] | torch.Tensor] = None,
        dtype: torch.dtype = torch.complex128,
        device: Optional[torch.device] = None,
        factors: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        self.n = n
        self.delays, self.dopplers, D, E = _grids(n, delays, dopplers, dtype, device)
        self.D, self.E = factors if factors is not None else (D, E)
        self.in_shape = (n,)
        self.out_shape = (len(self.delays), len(self.dopplers))

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.D.dtype)
        delayed = torch.fft.ifft(torch.fft.fft(x) * self.D, dim=-1)
        return _abs2((x * delayed.conj()) @ self.E.T)
