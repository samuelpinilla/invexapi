"""1D deblurring with the log invex penalty, via FISTA.

Recovers a sparse signal ``x_true`` from a blurred, noisy observation
``y = A x_true + noise`` (A = box blur), minimizing ``0.5*||Ax-y||^2 + lamb*g(x)``
where ``g`` is the log invex penalty.

Running this script still prints FISTA's UserWarning: the data-fidelity term alone
is certified convex, but the *combined* objective isn't (see
``invexapi.penalties.Sum`` - certificates never compose automatically).
"""

import torch

from invexapi import Certificate, Loss, LogInvexPenalty
from invexapi.optim import FISTA


def _blur_matrix(n: int, width: int = 5) -> torch.Tensor:
    A = torch.zeros(n, n)
    half = width // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        A[i, lo:hi] = 1.0 / (hi - lo)
    return A


class _DataFidelity(Loss):
    """0.5*||Ax-y||^2. Convex by inspection (A@x is linear), certified here to show
    a Loss outside invexapi.penalties declaring its own certificate."""

    def __init__(self, A: torch.Tensor, y: torch.Tensor):
        super().__init__()
        self.A = A
        self.y = y
        self._certify("convex", Certificate(status="assumed", source="manual"))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        r = self.A @ x - self.y
        return 0.5 * torch.sum(r * r)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return self.A.T @ (self.A @ x - self.y)


def main():
    torch.manual_seed(0)
    n = 200
    sparsity = 20

    x_true = torch.zeros(n)
    idx = torch.randperm(n)[:sparsity]
    x_true[idx] = torch.randn(sparsity) * 3.0

    A = _blur_matrix(n)
    noise = torch.randn(n) * 0.01
    y = A @ x_true + noise

    smooth = _DataFidelity(A, y)
    penalty = LogInvexPenalty(lamb=5e-3)

    L = torch.linalg.eigvalsh(A.T @ A).max().item()
    solver = FISTA(smooth, penalty, step=1.0 / L, max_iter=300, tol=1e-10)

    x_hat, history = solver.run(y.clone())

    blurred_mse = torch.mean((y - x_true) ** 2).item()
    deblurred_mse = torch.mean((x_hat - x_true) ** 2).item()

    print(f"objective history: {history[0]:.4f} -> {history[-1]:.4f}")
    print(f"MSE blurred:   {blurred_mse:.4f}")
    print(f"MSE deblurred: {deblurred_mse:.4f}")


if __name__ == "__main__":
    main()
