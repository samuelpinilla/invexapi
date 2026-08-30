import torch

from invexapi import QuasinormInvexPenalty
from invexapi.optim import LinearizedADMM
from invexapi.penalties.operators import FiniteDifference2D


class _Quadratic:
    def __init__(self, y: torch.Tensor):
        self.y = y

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum((x - self.y) ** 2)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.y


def test_admm_with_identity_operator_denoises_sparse_signal():
    torch.manual_seed(0)
    n = 100
    x_true = torch.zeros(n)
    idx = torch.randperm(n)[:10]
    x_true[idx] = torch.randn(10) * 3.0
    y = x_true + torch.randn(n) * 0.1

    smooth = _Quadratic(y)
    penalty = QuasinormInvexPenalty(lamb=0.05, q=0.5)
    solver = LinearizedADMM(smooth, penalty, rho=1.0, max_iter=300, tol=1e-10)

    x_hat, history = solver.run(y.clone())

    denoised_mse = torch.mean((x_hat - x_true) ** 2).item()
    noisy_mse = torch.mean((y - x_true) ** 2).item()
    assert denoised_mse < noisy_mse
    assert history[-1] <= history[0]


def test_admm_with_tv_operator_recovers_piecewise_constant_image():
    torch.manual_seed(0)
    x_true = torch.zeros(20, 20)
    x_true[5:15, 5:15] = 1.0
    y = x_true + torch.randn(20, 20) * 0.05

    smooth = _Quadratic(y)
    penalty = QuasinormInvexPenalty(lamb=0.02, q=0.5)
    solver = LinearizedADMM(
        smooth, penalty, D=FiniteDifference2D(), rho=1.0, max_iter=300, tol=1e-10
    )

    x_hat, _ = solver.run(y.clone())

    denoised_mse = torch.mean((x_hat - x_true) ** 2).item()
    noisy_mse = torch.mean((y - x_true) ** 2).item()
    assert denoised_mse < noisy_mse


def test_admm_project_is_applied():
    torch.manual_seed(0)
    y = torch.tensor([-1.0, 2.0, -3.0, 4.0])
    smooth = _Quadratic(y)
    penalty = QuasinormInvexPenalty(lamb=0.01, q=0.5)
    solver = LinearizedADMM(
        smooth, penalty, rho=1.0, project=lambda x: x.clamp(min=0.0), max_iter=50
    )

    x_hat, _ = solver.run(y.clamp(min=0.0))

    assert torch.all(x_hat >= 0.0)
