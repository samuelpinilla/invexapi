import numpy as np
import pytest
import torch

from invexapi import LLqPenalty
from invexapi.core import NumericalVerifier
from invexapi.functions.separable.llq import _LLqProx
from tests.reference_numpy import llq_prox_np

_CASES = [(0.1, 1.0, 0.5), (0.5, 2.0, 0.3), (1.0, 0.5, 0.9), (2.0, 1.0, 1.0), (0.3, 3.0, 0.7)]


def _brute_force(penalty, x, lamb):
    u = x.abs().numpy()
    grid = np.linspace(0.0, u.max() + 1.0, 400001)
    s = (lamb / penalty.beta) * np.log1p(penalty.beta * grid ** penalty.q)
    return np.sign(x.numpy()) * np.array([grid[np.argmin(s + 0.5 * (grid - ui) ** 2)] for ui in u])


@pytest.mark.parametrize("lamb,beta,q", _CASES)
def test_prox_is_global_minimizer(lamb, beta, q):
    penalty = LLqPenalty(lamb, q, beta)
    x = torch.linspace(-4, 4, 161, dtype=torch.float64)
    np.testing.assert_allclose(penalty.prox(x, 1.0).numpy(), _brute_force(penalty, x, lamb), atol=5e-5)


@pytest.mark.parametrize("lamb,beta,q", _CASES[:3])
def test_prox_matches_kernel_away_from_threshold(lamb, beta, q):
    # The kernel's bracket is exact once |x| is well above the threshold; near it, it
    # can return a local minimizer that loses to 0 (covered by the brute-force test).
    x = torch.linspace(-4, 4, 161, dtype=torch.float64)
    far = x.abs().numpy() > 3 * (lamb * q) ** (1 / (2 - q)) + 0.5
    y = LLqPenalty(lamb, q, beta).prox(x, 1.0).numpy()
    np.testing.assert_allclose(y[far], llq_prox_np(x.numpy(), lamb, beta, q)[far], atol=1e-5)


def test_step_scales_like_lamb():
    x = torch.linspace(-3, 3, 31, dtype=torch.float64)
    assert torch.allclose(LLqPenalty(0.2, 0.5).prox(x, 2.5), LLqPenalty(0.5, 0.5).prox(x, 1.0))


@pytest.mark.parametrize("lamb,beta,q", _CASES)
def test_backward_is_exact(lamb, beta, q):
    x = torch.tensor([-2.5, -1.7, -0.01, 0.02, 1.3, 2.9], dtype=torch.float64, requires_grad=True)
    lam = torch.tensor(lamb, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t, l: _LLqProx.apply(t, l, q, beta), (x, lam), eps=1e-6, atol=1e-5)


def test_prox_works_in_float32():
    x = torch.linspace(-3, 3, 61)
    y = LLqPenalty(0.1, 0.5).prox(x, 1.0)
    assert y.dtype == torch.float32 and torch.isfinite(y).all()
    np.testing.assert_allclose(y.numpy(), LLqPenalty(0.1, 0.5).prox(x.double(), 1.0).numpy(), atol=1e-5)


def test_value_and_grad_are_consistent():
    x = torch.tensor([-1.5, -0.3, 0.4, 2.0], dtype=torch.float64, requires_grad=True)
    penalty = LLqPenalty(0.3, 0.6, 2.0)
    penalty.value(x).backward()
    assert torch.allclose(x.grad, penalty.grad(x.detach()))


@pytest.mark.parametrize("kwargs", [{"lamb": 0.0, "q": 0.5}, {"lamb": 0.1, "q": 0.0}, {"lamb": 0.1, "q": 1.5}, {"lamb": 0.1, "q": 0.5, "beta": 0.0}])
def test_invalid_params_rejected(kwargs):
    with pytest.raises(ValueError):
        LLqPenalty(**kwargs)


def test_certified_admissible_hence_invex():
    penalty = LLqPenalty(0.1, 0.5, 2.0)
    assert penalty.admissible.reference.locator.startswith("Definition 4")
    assert penalty.invex.source == "rule:admissible_invex"
    assert penalty.convex is None
    assert NumericalVerifier().check(penalty, "admissible") is not None
