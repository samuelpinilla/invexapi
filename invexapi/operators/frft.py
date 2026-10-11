from __future__ import annotations

import math
from typing import Optional

import torch

from ..metadata import DesignDecision, Invariant, Provenance, Reference, documented
from .base import LinearOperator

__all__ = ["frft", "frft_matrix", "FractionalFourier"]

_OZAKTAS_REFERENCE = Reference(
    authors="Ozaktas, H. M., Arikan, O., Kutay, M. A. and Bozdagi, G.",
    title="Digital computation of the fractional Fourier transform",
    venue="IEEE Transactions on Signal Processing, 44(9)",
    year=1996,
    locator="Section III (chirp multiplication / convolution / multiplication)",
)


def _fconv(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Linear convolution along the last dim via a power-of-two FFT."""
    n = x.shape[-1] + y.shape[-1] - 1
    p = 1 << (n - 1).bit_length()
    z = torch.fft.ifft(torch.fft.fft(x, p, dim=-1) * torch.fft.fft(y, p, dim=-1), dim=-1)
    return z[..., :n]


def _sinc_interp(x: torch.Tensor) -> torch.Tensor:
    """Band-limited 2x upsampling: length N -> 2N-1."""
    n = x.shape[-1]
    y = torch.zeros(*x.shape[:-1], 2 * n - 1, dtype=x.dtype, device=x.device)
    y[..., ::2] = x
    t = torch.arange(-(2 * n - 3), 2 * n - 2, dtype=torch.float64, device=x.device) / 2
    xint = _fconv(y, torch.sinc(t).to(x.dtype))
    return xint[..., 2 * n - 3 : 4 * n - 4]


def _centered_dft(f: torch.Tensor, shft: torch.Tensor, inverse: bool) -> torch.Tensor:
    sn = math.sqrt(f.shape[-1])
    out = torch.empty_like(f)
    if inverse:
        out[..., shft] = torch.fft.ifft(f[..., shft], dim=-1) * sn
    else:
        out[..., shft] = torch.fft.fft(f[..., shft], dim=-1) / sn
    return out


@documented(
    provenance=[
        Provenance(
            description=(
                "the fast O(N log N) discrete FrFT: exact a in {0,1,2,3} cases, "
                "reduction of the order to 0.5 <= a <= 1.5, 2x sinc interpolation, "
                "chirp premultiplication, chirp convolution, chirp "
                "postmultiplication and 2x decimation"
            ),
            reference=_OZAKTAS_REFERENCE,
        )
    ],
    decisions=[
        DesignDecision(
            choice="transform along the last dim, batched over all leading dims",
            rejected="transforming along the first dim (column vectors)",
            rationale=(
                "PyTorch's fft convention is dim=-1; batching lets frft_matrix "
                "build the whole N x N transform from one call on eye(N)"
            ),
            reference=_OZAKTAS_REFERENCE,
        ),
        DesignDecision(
            choice="keep the widely used centering convention, including its even-N half-sample offset",
            rejected="re-centering the chirp grid on index N//2",
            rationale=(
                "for even N the general-order chirp grid is centered at (N-1)/2 "
                "while the a=1 DFT case and the order reductions are centered at "
                "N//2 (pinned by Hermite-Gauss eigenfunction tests: exact at odd "
                "N, and at even N when centered at (N-1)/2). Kept so results are "
                "comparable with the common reference implementation of this "
                "algorithm; any fixed linear operator serves lifted recovery equally"
            ),
        ),
    ],
)
def frft(f: torch.Tensor, a: float) -> torch.Tensor:
    """Fast discrete fractional Fourier transform of order ``a`` along the last dim.

    ``a = 1`` is the centered unitary DFT, ``a = 2`` time reversal, ``a = 0`` the
    identity; the transform is only approximately unitary/additive for general
    ``a`` (it samples the continuous FrFT).
    """
    if not f.is_complex():
        f = f.to(torch.complex128 if f.dtype == torch.float64 else torch.complex64)
    n = f.shape[-1]
    shft = (torch.arange(n, device=f.device) + n // 2) % n
    a = math.fmod(a, 4.0)
    if a < 0:
        a += 4.0

    if a == 0.0:
        return f.clone()
    if a == 2.0:
        return torch.flip(f, dims=(-1,))
    if a == 1.0:
        return _centered_dft(f, shft, inverse=False)
    if a == 3.0:
        return _centered_dft(f, shft, inverse=True)

    if a > 2.0:
        a -= 2.0
        f = torch.flip(f, dims=(-1,))
    if a > 1.5:
        a -= 1.0
        f = _centered_dft(f, shft, inverse=False)
    if a < 0.5:
        a += 1.0
        f = _centered_dft(f, shft, inverse=True)

    alpha = a * math.pi / 2
    tana2 = math.tan(alpha / 2)
    sina = math.sin(alpha)
    real = torch.float64 if f.dtype == torch.complex128 else torch.float32

    pad = torch.zeros(*f.shape[:-1], n - 1, dtype=f.dtype, device=f.device)
    f = torch.cat([pad, _sinc_interp(f), pad], dim=-1)

    grid = torch.arange(-2 * n + 2, 2 * n - 1, dtype=real, device=f.device)
    chrp = torch.exp(-1j * (math.pi / n * tana2 / 4) * grid**2)
    f = chrp * f

    c = math.pi / n / sina / 4
    kgrid = torch.arange(-(4 * n - 4), 4 * n - 3, dtype=real, device=f.device)
    faf = _fconv(torch.exp(1j * c * kgrid**2).to(f.dtype), f)
    faf = faf[..., 4 * n - 4 : 8 * n - 7] * math.sqrt(c / math.pi)
    faf = chrp * faf
    return complex(math.cos((1 - a) * math.pi / 4), -math.sin((1 - a) * math.pi / 4)) * faf[
        ..., n - 1 : 3 * n - 2 : 2
    ]


def frft_matrix(
    n: int,
    a: float,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """The ``n x n`` matrix ``F`` with ``F @ x == frft(x, a)``."""
    eye = torch.eye(n, dtype=dtype, device=device)
    return frft(eye, a).transpose(-1, -2).contiguous()


@documented(
    decisions=[
        DesignDecision(
            choice="materialize the FrFT matrix once and use its exact conjugate transpose as the adjoint",
            rejected="using frft(v, -a) as the adjoint",
            rationale=(
                "the discrete fast FrFT is only approximately unitary, so "
                "frft(., -a) fails the inner-product adjoint identity that "
                "FASTA's backtracking and BB step sizes rely on; N x N is cheap "
                "at radar pulse lengths"
            ),
        )
    ],
    invariants=[Invariant("adjoint(v) == matrix.conj().T @ v exactly, not frft(v, -a).")],
)
class FractionalFourier(LinearOperator):
    """``x -> frft(x, a)`` along the last dim, as an explicit ``n x n`` matrix."""

    def __init__(
        self,
        n: int,
        a: float,
        dtype: torch.dtype = torch.complex128,
        device: Optional[torch.device] = None,
    ):
        self.n = n
        self.a = a
        self.matrix = frft_matrix(n, a, dtype=dtype, device=device)
        self.in_shape = self.out_shape = (n,)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.matrix.dtype) @ self.matrix.T

    def adjoint(self, v: torch.Tensor) -> torch.Tensor:
        return v.to(self.matrix.dtype) @ self.matrix.conj()
