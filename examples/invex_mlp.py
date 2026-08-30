"""Fit a synthetic nonlinear function: invex-certified MLP vs. a plain MLP."""

import torch
import torch.nn as nn

from invexapi.nn import ixLinear, ixSequential


def build_invex_model() -> nn.Module:
    return ixSequential(
        ixLinear(in_features=2, hidden_features=32, out_features=16, activation=nn.Tanh()),
        ixLinear(in_features=16, hidden_features=16, out_features=1, activation=nn.Tanh()),
    )


def build_plain_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(2, 32),
        nn.Tanh(),
        nn.Linear(32, 16),
        nn.Tanh(),
        nn.Linear(16, 16),
        nn.Tanh(),
        nn.Linear(16, 1),
    )


def train(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int = 200) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def main():
    torch.manual_seed(0)
    n = 512
    x = torch.rand(n, 2) * 4 - 2
    y = (torch.sin(x[:, 0]) + x[:, 1] ** 2).unsqueeze(1)

    torch.manual_seed(1)
    invex_model = build_invex_model()
    print(f"invex_model.invex: {invex_model.invex}")
    invex_losses = train(invex_model, x, y)

    torch.manual_seed(1)
    plain_model = build_plain_model()
    print(f"plain_model has no .invex attribute: {not hasattr(plain_model, 'invex')}")
    plain_losses = train(plain_model, x, y)

    print(f"{'epoch':>6}  {'invex loss':>12}  {'plain loss':>12}")
    for epoch in [1, 50, 100, 150, 200]:
        i = epoch - 1
        print(f"{epoch:6d}  {invex_losses[i]:12.4f}  {plain_losses[i]:12.4f}")


if __name__ == "__main__":
    main()
