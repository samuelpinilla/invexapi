from __future__ import annotations

from typing import Optional, Sequence

import torch

from ..metadata import DesignDecision, Invariant, Provenance, Reference, documented
from .base import LinearOperator, NonlinearOperator
from .radar_ambiguity import integer_grid

__all__ = ["FROG", "LiftedFROG"]

_FROG_REFERENCE = Reference(
    authors="Pinilla, S., Bendory, T., Eldar, Y. C. & Arguello, H.",
    title="Frequency-resolved optical gating recovery via smoothing gradient",
    venue="IEEE Transactions on Signal Processing, 67(23), 6121-6132",
    year=2019,
    locator="Eq. (1)",
)


def _grids(
    n: int,
    delays: Optional[Sequence[float] | torch.Tensor],
    frequencies: Optional[Sequence[float] | torch.Tensor],
    dtype: torch.dtype,
    device: Optional[torch.device],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Delay/frequency grids plus the factors ``D[tau, l] = e^{+2 pi i l tau / n}``
    (centred frequencies ``l in [-n/2, n/2)``, an advance ``x[n + tau]``) and
    ``E[k, m] = e^{-2 pi i m k / n}``."""
    delay_grid = integer_grid(n) if delays is None else torch.as_tensor(delays, dtype=torch.float64)
    frequencies_grid = integer_grid(n) if frequencies is None else torch.as_tensor(frequencies, dtype=torch.float64)
    freqs = torch.fft.fftfreq(n, d=1.0 / n, dtype=torch.float64)
    time = torch.arange(n, dtype=torch.float64)
    D = torch.exp(2j * torch.pi * torch.outer(delay_grid, freqs) / n).to(dtype=dtype, device=device)
    E = torch.exp(-2j * torch.pi * torch.outer(frequencies_grid, time) / n).to(dtype=dtype, device=device)
    return delay_grid, frequencies_grid, D, E


def _abs2(z: torch.Tensor) -> torch.Tensor:
    # |z|^2 as real^2 + imag^2: smooth everywhere, so forward-mode Jacobians stay
    # exact at trace samples that vanish
    return z.real**2 + z.imag**2


@documented(
    provenance=[
        Provenance(
            description=(
                "the lifted measurement Tr(B_{tau,k} X) with B_{tau,k} = Phi_tau^T diag(e^{-2 pi i n k / N}), "
                "whose magnitude squared at X = x x^T (no conjugate) is the second-harmonic-generation "
                "FROG trace Z[p, k] = |sum_n x[n] x[n + pL] e^{-2 pi i n k / N}|^2"
            ),
            reference=_FROG_REFERENCE,
        )
    ],
    decisions=[
        DesignDecision(
            choice=(
                "delays as a Fourier-domain phase ramp Phi_tau = F^H diag(e^{+2 pi i l tau/N}) F "
                "(centred l), on an arbitrary real delay grid"
            ),
            rejected="integer cyclic index shifts x[n + pL] only",
            rationale=(
                "the ramp reproduces Eq. (1) exactly on the grid tau = pL, extends it to "
                "fractional delays, and makes incomplete traces (L > 1, or any delay set) "
                "just a choice of grid"
            ),
        ),
        DesignDecision(
            choice="the domain is all (N, N) matrices; symmetry of X is left to the penalty",
            rejected="restricting apply/adjoint to symmetric matrices",
            rationale=(
                "keeps adjoint the exact conjugate transpose of apply on C^{N x N}; a "
                "penalty whose prox maps onto symmetric matrices then keeps every "
                "forward-backward iterate symmetric"
            ),
        ),
        DesignDecision(
            choice="matrix-free: fft of X along rows, one (N, N) x (N, T) and one (K, N) x (N, T) matmul",
            rejected="materializing the T*K matrices B_{tau,k}",
            rationale="the factored form costs O(N^2 (T + K)) per application",
        ),
    ],
    invariants=[
        Invariant(
            "apply(x x^T)[p, k] == sum_n x[n] x[n + p] e^{-2 pi i n k / N} for integer p, k, so "
            "|apply(x x^T)|^2 is the FROG trace of Eq. (1) on the delay grid p*L."
        ),
        Invariant("adjoint is the exact conjugate transpose of apply under Re<U, V>."),
        Invariant("on the full integer grid (T = K = N), adjoint(apply(X)) == N * X."),
    ],
)
class LiftedFROG(LinearOperator):
    """Lifted FROG measurement ``X -> Tr(B_{tau,k} X)``.

    ``apply`` maps an ``(N, N)`` matrix ``X`` to a ``(T, K)`` tensor indexed
    ``[tau, k]``; for ``X = x x^T`` (transpose, not conjugate transpose) its
    magnitude squared is the SHG FROG trace
    ``Z[tau, k] = |sum_n x[n] (Phi_tau x)[n] e^{-2 pi i n k / N}|^2``, where
    ``Phi_tau`` advances by ``tau`` samples through a Fourier-domain phase ramp.
    ``delays`` and ``frequencies`` default to the integers ``0..N-1``
    (``L = 1``); ``integer_grid(N, L)`` gives the decimated delays ``pL``.
    """

    def __init__(
        self,
        n: int,
        delays: Optional[Sequence[float] | torch.Tensor] = None,
        frequencies: Optional[Sequence[float] | torch.Tensor] = None,
        dtype: torch.dtype = torch.complex128,
        device: Optional[torch.device] = None,
    ):
        self.n = n
        self.delays, self.frequencies, self.D, self.E = _grids(n, delays, frequencies, dtype, device)
        # P[m, l] = e^{+2 pi i l m / n} / n: undoes the row fft at the diagonal m
        self.P = torch.fft.ifft(torch.eye(n, dtype=dtype, device=device), dim=0)
        self.in_shape = (n, n)
        self.out_shape = (len(self.delays), len(self.frequencies))

    @property
    def shape(self):
        """Measurement shape ``(T, K)``."""
        return self.out_shape

    def delay(self, x: torch.Tensor) -> torch.Tensor:
        """``(T, N)`` advanced copies ``Phi_tau x`` of a signal ``x``."""
        return torch.fft.ifft(torch.fft.fft(x.to(self.D.dtype)) * self.D, dim=-1)

    def measure(self, x: torch.Tensor) -> torch.Tensor:
        """``apply(x x^T)`` computed from the signal directly, in O(T N (log N + K))."""
        g = x.to(self.D.dtype) * self.delay(x)
        return g @ self.E.T

    def trace(self, x: torch.Tensor) -> torch.Tensor:
        """The FROG trace ``Z[tau, k] = |apply(x x^T)|^2``."""
        return _abs2(self.measure(x))

    def signal_operator(self) -> "FROG":
        """The nonlinear map ``x -> Z[tau, k]`` on signals, sharing these grids."""
        return FROG(self.n, self.delays, self.frequencies, factors=(self.D, self.E))

    def apply(self, X: torch.Tensor) -> torch.Tensor:
        # (X Phi_tau^T)[m, m] = sum_l fft(X)[m, l] P[m, l] D[tau, l]
        M = torch.fft.fft(X.to(self.D.dtype), dim=-1) * self.P
        G = M @ self.D.T
        return (self.E @ G).T

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        G = self.E.conj().T @ y.to(self.D.dtype).T
        M = (G @ self.D.conj()) * self.P.conj()
        return torch.fft.ifft(M, dim=-1) * self.n


@documented(
    decisions=[
        DesignDecision(
            choice="fit intensities Z[tau, k] (a quartic polynomial in x), not magnitudes sqrt(Z)",
            rejected="the magnitude map x -> |Tr(B (x x^T))|",
            rationale=(
                "|z| is not differentiable at z = 0, and many trace samples are near "
                "zero; the intensity residual is a polynomial, so Gauss-Newton keeps "
                "its local rate"
            ),
        )
    ],
)
class FROG(NonlinearOperator):
    """The SHG FROG trace of a signal, ``x -> Z[tau, k]``.

    ``Z[tau, k] = |sum_n x[n] (Phi_tau x)[n] e^{-2 pi i n k / N}|^2``, a ``(T, K)``
    real tensor; the signal-domain (nonlinear) counterpart of :class:`LiftedFROG`,
    whose ``apply(x x^T)`` it squares in magnitude.
    """

    def __init__(
        self,
        n: int,
        delays: Optional[Sequence[float] | torch.Tensor] = None,
        frequencies: Optional[Sequence[float] | torch.Tensor] = None,
        dtype: torch.dtype = torch.complex128,
        device: Optional[torch.device] = None,
        factors: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        self.n = n
        self.delays, self.frequencies, D, E = _grids(n, delays, frequencies, dtype, device)
        self.D, self.E = factors if factors is not None else (D, E)
        self.in_shape = (n,)
        self.out_shape = (len(self.delays), len(self.frequencies))

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.D.dtype)
        advanced = torch.fft.ifft(torch.fft.fft(x) * self.D, dim=-1)
        return _abs2((x * advanced) @ self.E.T)
