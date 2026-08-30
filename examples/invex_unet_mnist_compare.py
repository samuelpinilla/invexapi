"""InvexUNet vs. NormalUNet: architecture comparison on real MNIST denoising.

Trains both across several activations and random seeds, reporting the spread
of final losses. Needs ``torchvision``; downloads MNIST on first run. Runs on
CUDA automatically if available (``--device cpu`` to force CPU).
"""

import argparse
import statistics

import torch
import torch.nn as nn
import torchvision

from invex_unet import InvexUNet, NormalUNet
from _data import DATA_DIR

EPOCHS = 150
N_IMAGES = 64

ARCHITECTURES = {"InvexUNet": InvexUNet, "NormalUNet": NormalUNet}
ACTIVATIONS = {
    "Tanh": nn.Tanh,
    "Softplus": nn.Softplus,
    "ReLU": nn.ReLU,
}


def load_mnist_batch() -> torch.Tensor:
    dataset = torchvision.datasets.MNIST(root=str(DATA_DIR / "mnist"), train=True, download=True)
    images = dataset.data[:N_IMAGES].float() / 255.0  # [N, 28, 28] in [0, 1]
    return images.unsqueeze(1)  # [N, 1, 28, 28]


def train_once(
    model_cls,
    activation_cls,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    seed: int,
    device: torch.device,
    base_channels: int,
) -> float:
    torch.manual_seed(seed)
    model = model_cls(
        in_channels=1, out_channels=1, base_channels=base_channels, activation=activation_cls()
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for _ in range(EPOCHS):
        optimizer.zero_grad()
        loss = loss_fn(model(noisy), clean)
        loss.backward()
        optimizer.step()

    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--noise-std", type=float, default=0.4)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--num-seeds", type=int, default=20)
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeds = list(range(args.num_seeds))
    print(
        f"device: {device}  noise_std: {args.noise_std}  base_channels: {args.base_channels}  "
        f"num_seeds: {args.num_seeds}"
    )

    torch.manual_seed(0)
    clean = load_mnist_batch().to(device)
    noisy = (clean + torch.randn_like(clean) * args.noise_std).to(device)

    header = f"{'architecture':12s} {'activation':10s} {'final losses (per seed)':40s} {'mean':>8s} {'std':>8s}"
    print(header)
    for arch_name, model_cls in ARCHITECTURES.items():
        for act_name, activation_cls in ACTIVATIONS.items():
            losses = [
                train_once(model_cls, activation_cls, clean, noisy, seed, device, args.base_channels)
                for seed in seeds
            ]
            mean = statistics.mean(losses)
            std = statistics.pstdev(losses)
            losses_str = ", ".join(f"{l:.4f}" for l in losses)
            print(f"{arch_name:12s} {act_name:10s} {losses_str:40s} {mean:8.4f} {std:8.4f}")

    print("\nLower std = more consistent convergence across seeds; lower mean = better fit.")


if __name__ == "__main__":
    main()
