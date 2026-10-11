import pytest
import torch

from invexapi import L1Penalty, PSDTracePenalty
from invexapi.operators import LinearOperator
from invexapi.optim import FASTA


class _Matrix(LinearOperator):
    def __init__(self, M):
        self.M = M

    def apply(self, x):
        return self.M @ x

    def adjoint(self, v):
        return self.M.conj().T @ v


class _LeastSquares:
    def __init__(self, b):
        self.b = b

    def value(self, z):
        return 0.5 * (z - self.b).abs().pow(2).sum()

    def grad(self, z):
        return z - self.b


def _lasso(seed=0):
    g = torch.Generator().manual_seed(seed)
    m, n = 40, 60
    M = torch.randn(m, n, dtype=torch.float64, generator=g)
    x_true = torch.zeros(n, dtype=torch.float64)
    x_true[torch.randperm(n, generator=g)[:5]] = torch.randn(5, dtype=torch.float64, generator=g) * 3
    return M, M @ x_true, x_true


def _kkt_violation(M, b, lamb, x):
    # 0 in M^T(Mx-b) + lamb*d||x||_1
    g = M.T @ (M @ x - b)
    on = x.abs() > 1e-8
    v_on = (g[on] + lamb * torch.sign(x[on])).abs().max()
    v_off = torch.clamp(g[~on].abs() - lamb, min=0).max() if (~on).any() else torch.tensor(0.0)
    return max(v_on.item(), v_off.item())


@pytest.mark.parametrize(
    "kwargs",
    [dict(), dict(accelerate=True), dict(adaptive=False), dict(accelerate=True, restart=False)],
)
def test_fasta_solves_lasso_in_every_mode(kwargs):
    M, b, _ = _lasso()
    lamb = 0.5
    solver = FASTA(
        _LeastSquares(b), L1Penalty(lamb), A=_Matrix(M), max_iter=5000, tol=1e-10,
        record_objective=True, generator=torch.Generator().manual_seed(0), **kwargs,
    )
    with pytest.warns(UserWarning, match="smooth"):
        x, history = solver.run(torch.zeros(M.shape[1], dtype=torch.float64))
    assert _kkt_violation(M, b, lamb, x) < 1e-4
    assert history[-1] <= history[0]
    assert solver.info["iterations"] == len(solver.info["residuals"])


def test_fasta_handles_complex_lifted_problems():
    # min 0.5||X - Y||_F^2 + indicator_PSD: the answer is Y's PSD projection.
    g = torch.Generator().manual_seed(1)
    Y = torch.randn(5, 5, dtype=torch.complex128, generator=g)
    penalty = PSDTracePenalty(lamb=0.0)
    solver = FASTA(_LeastSquares(Y), penalty, max_iter=500, tol=1e-12, generator=g)
    with pytest.warns(UserWarning, match="smooth"):
        X, _ = solver.run(torch.zeros_like(Y))
    assert torch.allclose(X, penalty.prox(Y, 1.0), atol=1e-8)


def test_fasta_custom_stop_hook_and_validation():
    M, b, _ = _lasso(1)
    calls = []

    def stop_now(x, i, resid, norm_resid):
        calls.append(i)
        return i >= 3

    solver = FASTA(_LeastSquares(b), L1Penalty(0.1), A=_Matrix(M), stop_now=stop_now)
    with pytest.warns(UserWarning):
        solver.run(torch.zeros(M.shape[1], dtype=torch.float64))
    assert calls == [1, 2, 3]
    with pytest.raises(ValueError):
        FASTA(_LeastSquares(b), L1Penalty(0.1), stop_rule="nope")
