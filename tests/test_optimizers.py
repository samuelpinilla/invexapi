import pytest
import torch

from invexapi import LogInvexPenalty, QuasinormInvexPenalty
from invexapi.optim import FISTA, GradientDescent, NonlinearCG

# _Quadratic below carries no certificate, and QuasinormInvexPenalty/LogInvexPenalty
# only certify `invex` (never re-derived for the combined FISTA objective, see
# invexapi.penalties.Sum) - every solver.run() call in this file is expected to warn
# that no global-optimum guarantee applies, which these tests assert explicitly
# rather than letting print to stderr silently.


class _Quadratic:
    """0.5*||Ax-b||^2, a smooth strongly-convex toy objective."""

    def __init__(self, A: torch.Tensor, b: torch.Tensor):
        self.A = A
        self.b = b

    def value(self, x: torch.Tensor) -> torch.Tensor:
        r = self.A @ x - self.b
        return 0.5 * torch.sum(r * r)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return self.A.T @ (self.A @ x - self.b)


def _toy_problem():
    torch.manual_seed(0)
    n = 10
    A = torch.eye(n, dtype=torch.double) * 2.0
    x_star = torch.randn(n, dtype=torch.double)
    b = A @ x_star
    return A, b, x_star


def test_gradient_descent_converges_on_quadratic():
    A, b, x_star = _toy_problem()
    objective = _Quadratic(A, b)
    solver = GradientDescent(objective, step=0.2, max_iter=200, tol=1e-10)

    x0 = torch.zeros_like(x_star)
    with pytest.warns(UserWarning, match="certificate"):
        x, history = solver.run(x0)

    assert torch.allclose(x, x_star, atol=1e-3)
    assert history[-1] < history[0]


def test_nonlinear_cg_converges_on_quadratic():
    A, b, x_star = _toy_problem()
    objective = _Quadratic(A, b)
    solver = NonlinearCG(objective, max_iter=50, tol=1e-10)

    x0 = torch.zeros_like(x_star)
    with pytest.warns(UserWarning, match="certificate"):
        x, history = solver.run(x0)

    assert torch.allclose(x, x_star, atol=1e-3)
    assert history[-1] < history[0]


def test_fista_converges_with_quasinorm_penalty():
    A, b, x_star = _toy_problem()
    smooth = _Quadratic(A, b)
    penalty = QuasinormInvexPenalty(lamb=1e-4, q=0.5)
    L = torch.linalg.eigvalsh(A.T @ A).max().item()
    solver = FISTA(smooth, penalty, step=1.0 / L, max_iter=200, tol=1e-10)

    x0 = torch.zeros_like(x_star)
    with pytest.warns(UserWarning, match="certificate"):
        x, history = solver.run(x0)

    # Small penalty weight -> solution should stay close to the unpenalized optimum.
    assert torch.allclose(x, x_star, atol=1e-2)
    assert history[-1] <= history[0]


def test_fista_converges_with_log_penalty():
    A, b, x_star = _toy_problem()
    smooth = _Quadratic(A, b)
    penalty = LogInvexPenalty(lamb=1e-4)
    L = torch.linalg.eigvalsh(A.T @ A).max().item()
    solver = FISTA(smooth, penalty, step=1.0 / L, max_iter=200, tol=1e-10)

    x0 = torch.zeros_like(x_star)
    with pytest.warns(UserWarning, match="certificate"):
        x, history = solver.run(x0)

    assert torch.allclose(x, x_star, atol=1e-2)
    assert history[-1] <= history[0]
