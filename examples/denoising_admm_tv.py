"""Total-variation image denoising via LinearizedADMM, on a real test image.

Adapted from ``code_ICLR/ADMM/ADMM_Lq.py``: TV-regularized denoising
(``min_x 0.5*||x-y||^2 + lamb*sum(|Dx|^q)``) where ``D`` is the 2D finite-difference
(TV) operator and the penalty is the quasinorm invex penalty already implemented in
this library — same configuration as the reference script, run here with
``LinearizedADMM`` + ``FiniteDifference2D`` + ``QuasinormInvexPenalty`` instead of
the reference's cupy/CUDA kernel, on ``examples/data/images/noisy_image.tif``
instead of the reference's own test image.

Running this script prints FISTA/GD/CG's usual UserWarning-style caveat: `smooth`
here (this example's own `_DataFidelity`) is uncertified, so LinearizedADMM warns
that no global-optimum guarantee applies — expected, same as the other examples.

Runs on CUDA automatically if available (``python examples/denoising_admm_tv.py``),
or force CPU with ``--device cpu``. Nothing in invexapi is device-specific — every
op inherits its device from its input tensors — so the only thing this script does
for GPU support is move the image tensor to the target device before construction;
everything downstream (the penalty's prox, the TV operator, the solver) follows.
"""

import argparse
import time

import matplotlib.pyplot as plt
import torch

from invexapi import Loss, QuasinormInvexPenalty
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
    penalty = QuasinormInvexPenalty(lamb=1.1e-1, q=0.85)
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
