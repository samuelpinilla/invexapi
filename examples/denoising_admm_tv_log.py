"""Total-variation image denoising via LinearizedADMM, using the log penalty.

Same setup as ``denoising_admm_tv.py`` (``min_x 0.5*||x-y||^2 + lamb*g(Dx)`` via
``LinearizedADMM`` + ``FiniteDifference2D`` on
``examples/data/images/noisy_image.tif``), but with ``g`` = ``LogInvexPenalty``
instead of ``QuasinormInvexPenalty`` - the reference ADMM scripts only pair TV with
the quasinorm/L1/Llq penalties, so there is no original CUDA kernel to validate
this specific combination against; it demonstrates that any `Penalty` (paper-sourced
or not) works as `LinearizedADMM`'s second argument, per its generic `penalty.prox`
contract.

Runs on CUDA automatically if available, or force CPU with ``--device cpu`` (see
``denoising_admm_tv.py``'s docstring for why no other device-handling is needed).
"""

import argparse
import time

import matplotlib.pyplot as plt
import torch

from invexapi import Loss, LogInvexPenalty
from invexapi.optim import LinearizedADMM
from invexapi.penalties.operators import FiniteDifference2D

from _data import load_image


class _DataFidelity(Loss):
    def __init__(self, y: torch.Tensor):
        super().__init__()
        self.y = y

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum((x - self.y) ** 2)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device to run on, e.g. 'cuda', 'cuda:1', 'cpu' (default: cuda if available)",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    y = load_image("images/noisy_image.tif").to(device)

    print(f"running on: {device}")

    smooth = _DataFidelity(y)
    penalty = LogInvexPenalty(lamb=3e-1)
    solver = LinearizedADMM(
        smooth,
        penalty,
        D=FiniteDifference2D(),
        rho=1.6,
        project=lambda x: x.clamp(min=0.0),
        max_iter=200,
        tol=1e-8,
    )

    start_time = time.time()
    x_hat, history = solver.run(y.clone())
    end_time = time.time()

    print(f"image shape: {tuple(y.shape)}")
    print(f"objective history: {history[0]:.4f} -> {history[-1]:.4f}")
    print(f"denoised range: [{x_hat.min().item():.4f}, {x_hat.max().item():.4f}]")
    print(f"mean |x_hat - y|: {torch.mean((x_hat - y).abs()).item():.6f}")
    print(f"elapsed time: {end_time - start_time:.4f} seconds")

    plt.imshow(x_hat.cpu().numpy(), cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
