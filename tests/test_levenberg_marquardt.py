import pytest
import torch

from invexapi.operators import LiftedFrFTAmbiguity, NonlinearOperator, frft_orders
from invexapi.optim import LevenbergMarquardt


class _Exponential(NonlinearOperator):
    """theta -> a * exp(-k t) on a fixed grid t."""

    def __init__(self, t):
        self.t = t

    def apply(self, theta):
        return theta[0] * torch.exp(-theta[1] * self.t)


class _ComplexSquare(NonlinearOperator):
    """z -> (M z) * (M z), elementwise (holomorphic, complex in and out)."""

    def __init__(self, M):
        self.M = M

    def apply(self, z):
        w = self.M @ z
        return w * w


def test_lm_fits_a_real_exponential_to_machine_precision():
    t = torch.linspace(0, 4, 30, dtype=torch.float64)
    op = _Exponential(t)
    y = op(torch.tensor([2.0, 1.3], dtype=torch.float64))
    solver = LevenbergMarquardt(op, y, max_iter=100)
    with pytest.warns(UserWarning, match="least-squares"):
        theta, history = solver.run(torch.tensor([1.0, 0.5], dtype=torch.float64))
    assert torch.allclose(theta, torch.tensor([2.0, 1.3], dtype=torch.float64), atol=1e-10)
    assert history[-1] < 1e-12 and history == sorted(history, reverse=True)


def test_lm_handles_complex_unknowns_and_residuals():
    g = torch.Generator().manual_seed(0)
    M = torch.randn(20, 5, dtype=torch.complex128, generator=g)
    z_true = torch.randn(5, dtype=torch.complex128, generator=g)
    op = _ComplexSquare(M)
    start = z_true + 0.05 * torch.randn(5, dtype=torch.complex128, generator=g)
    with pytest.warns(UserWarning):
        z, _ = LevenbergMarquardt(op, op(z_true), max_iter=100).run(start)
    # z -> -z is the only ambiguity of an elementwise square
    assert min((z - z_true).norm(), (z + z_true).norm()) < 1e-10


def test_lm_recovers_a_signal_from_its_frft_ambiguity_function_near_the_truth():
    g = torch.Generator().manual_seed(1)
    n = 15
    op = LiftedFrFTAmbiguity(n, frft_orders(n, 1)).signal_operator()
    x = torch.randn(n, dtype=torch.complex128, generator=g)
    start = x + 0.01 * torch.randn(n, dtype=torch.complex128, generator=g)
    solver = LevenbergMarquardt(op, op(x), max_iter=200)
    with pytest.warns(UserWarning):
        q, history = solver.run(start)
    assert history[-1] < 1e-9
    phase = torch.vdot(q, x)
    assert (x - phase / phase.abs() * q).norm() / x.norm() < 1e-4
