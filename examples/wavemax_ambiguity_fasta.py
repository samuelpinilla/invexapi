"""Lifted radar-waveform recovery from a FrFT-based ambiguity function, via FASTA.

A compact version of the WaveMax method (Pinilla et al., IEEE TSP 2025, problem
(14)): given only the ambiguity function ``A = |B(x x^H)|^2`` of a pulse ``x``,
recover the lifted matrix ``Z ~ x x^H`` by minimizing

    0.5*|| |B(Z)| + s - sqrt(A) ||^2  -  mu*Re<X0, Z>  +  mu*tr(Z)  +  i_PSD(Z)

with FASTA, where ``X0 = x0 x0^H`` is an anchor and ``s`` a slack refreshed
between rounds. Every piece is an invexapi building block:

- ``LiftedFrFTAmbiguity``                    the linear map ``Z -> Tr(B_{alpha,k} Z)``
- ``SlackAmplitudeLeastSquares``             the smooth term (nonconvex: uncertified)
- ``LinearTilt(PSDTracePenalty(mu), mu*X0)`` the proximal term (PSD projection)
- ``FASTA``                                  adaptive forward-backward splitting

For brevity the anchor here is a perturbed copy of the true pulse, and the
error is measured up to a global phase only; a full pipeline would build the
anchor by spectral initialization (the paper's Algorithm 1) and measure error up
to all of the ambiguity function's trivial ambiguities (its Eq. (11)).
Expect FASTA's certificate warnings.
"""

import math

import torch

from invexapi import SlackAmplitudeLeastSquares, LinearTilt, PSDTracePenalty
from invexapi.operators import LiftedFrFTAmbiguity, frft_orders
from invexapi.optim import FASTA


def _chirp_pulse(n: int) -> torch.Tensor:
    """A Gaussian-windowed LFM pulse, unit norm."""
    t = torch.arange(n, dtype=torch.float64) - n / 2
    envelope = torch.exp(-((t / (n / 8)) ** 2))
    x = envelope * torch.exp(1j * math.pi * 0.02 * t**2)
    return x / x.norm()


def _phase_aligned_error(x: torch.Tensor, q: torch.Tensor) -> float:
    """``min_beta ||q - e^{i beta} x|| / ||x||`` (global phase only)."""
    inner = torch.vdot(x, q)
    return (q - inner / inner.abs() * x).norm().item() / x.norm().item()


def main():
    torch.manual_seed(0)
    n = 48
    x = _chirp_pulse(n)

    op = LiftedFrFTAmbiguity(n, frft_orders(n, decimation=2))
    b = op.measure(x).abs()  # sqrt of the ambiguity function

    x0 = x + 0.15 * torch.randn(n, dtype=torch.complex128) / math.sqrt(n)
    X0 = torch.outer(x0, x0.conj())
    mu = 1e-4 * torch.linalg.eigvals(op.adjoint(b)).abs().sum().item()
    penalty = LinearTilt(PSDTracePenalty(lamb=mu), mu * X0)

    Z, slack = X0.clone(), torch.zeros_like(b)
    for r in range(3):
        solver = FASTA(SlackAmplitudeLeastSquares(b, slack), penalty, A=op, max_iter=300, tol=1e-6)
        Z, _ = solver.run(Z)
        slack = torch.clamp(b - op.apply(Z).abs(), min=0.0)
        print(f"round {r + 1}: {solver.info['iterations']} iterations, "
              f"residual {solver.info['residuals'][-1]:.2e}")

    d, V = torch.linalg.eigh(0.5 * (Z + Z.conj().T))
    x_hat = d[-1].sqrt() * V[:, -1]
    print(f"anchor error         {_phase_aligned_error(x, x0):.4f}")
    print(f"reconstruction error {_phase_aligned_error(x, x_hat):.4f}")
    print(f"rank-1 energy ratio  {(d[-1] / d.clamp(min=0).sum()).item():.4f}")


if __name__ == "__main__":
    main()
