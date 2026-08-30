"""Sparse-signal denoising with the quasinorm invex penalty.

Adapted from the fixed-point structure of ``codes NEURIPS/denoising/denoisingEq6.py``
(dictionary-learning + patchwise invex filtering), simplified to a single synthetic
sparse signal so the example is self-contained and needs no external image files or
extra dependencies beyond ``invexapi`` itself.

Problem: recover a sparse ``x_true`` from noisy observations ``y = x_true + noise``
by minimizing ``0.5*||x-y||^2 + lamb*|x|^q`` with FISTA — the invex quasinorm prox
plays the role of the denoiser (like the paper's per-patch ``invex2DFilter`` step).

Running this script prints a UserWarning: nobody has certified the combined
data-fidelity+penalty objective as convex/invex (see
``invexapi.penalties.Sum``), so FISTA can't claim a global-optimum guarantee here —
that's the intended, honest behavior, not a bug.
"""

import torch

from invexapi import Loss, QuasinormInvexPenalty
from invexapi.optim import FISTA


class _DataFidelity(Loss):
    """0.5*||x-y||^2. Left uncertified on purpose (see module docstring) — running
    this example demonstrates FISTA's warning firing for an objective nobody has
    proven a global-optimum guarantee for."""

    def __init__(self, y: torch.Tensor):
        super().__init__()
        self.y = y

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum((x - self.y) ** 2)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.y


def main():
    torch.manual_seed(0)
    n = 200
    sparsity = 20

    x_true = torch.zeros(n)
    idx = torch.randperm(n)[:sparsity]
    x_true[idx] = torch.randn(sparsity) * 3.0

    noise = torch.randn(n) * 0.2
    y = x_true + noise

    smooth = _DataFidelity(y)
    penalty = QuasinormInvexPenalty(lamb=0.3, q=0.5)
    solver = FISTA(smooth, penalty, step=1.0, max_iter=200, tol=1e-8)

    x_hat, history = solver.run(y.clone())

    noisy_mse = torch.mean((y - x_true) ** 2).item()
    denoised_mse = torch.mean((x_hat - x_true) ** 2).item()

    print(f"objective history: {history[0]:.4f} -> {history[-1]:.4f}")
    print(f"MSE noisy:    {noisy_mse:.4f}")
    print(f"MSE denoised: {denoised_mse:.4f}")


if __name__ == "__main__":
    main()
