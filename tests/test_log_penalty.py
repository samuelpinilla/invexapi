import numpy as np
import pytest
import torch

from invexapi import LogPenalty
from tests.reference_numpy import log_prox_np


def brute_force_prox(penalty, x, lamb_step):
    """argmin_z lamb_step*phi(|z|) + 0.5*(z-x)^2 by dense grid search, per entry."""
    t = x.abs().numpy()
    grid = np.linspace(0.0, t.max() + 1.0, 200001)
    phi = penalty._phi(torch.from_numpy(grid)).numpy() * (lamb_step / penalty.lamb)
    out = np.empty_like(t)
    for i, ti in enumerate(t):
        out[i] = grid[np.argmin(penalty.lamb * phi + 0.5 * (grid - ti) ** 2)]
    return np.sign(x.numpy()) * out


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("lamb_ref", [1e-3, 0.05, 0.4])
def test_prox_matches_numpy_reference(lamb_ref, beta):
    # The paper's code penalizes (lamb/beta)*log(1+beta|x|); only compared where its
    # subproblem is convex (lamb_ref*beta <= 1), the regime where its clamp is exact.
    torch.manual_seed(0)
    x = (torch.rand(64) * 4 - 2).double()

    penalty = LogPenalty(lamb=lamb_ref / beta, beta=beta)
    y_torch = penalty.prox(x, step=1.0).numpy()

    y_ref = log_prox_np(x.numpy(), lamb_ref, beta)

    np.testing.assert_allclose(y_torch, y_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("lamb,beta", [(0.3, 1.0), (2.0, 1.0), (1.5, 3.0)])  # last two: nonconvex prox
def test_prox_is_global_minimizer(lamb, beta):
    penalty = LogPenalty(lamb=lamb, beta=beta)
    x = torch.linspace(-6, 6, 97, dtype=torch.double)

    y = penalty.prox(x, step=1.0)

    np.testing.assert_allclose(y.numpy(), brute_force_prox(penalty, x, lamb), atol=2e-3)


def test_step_scales_like_lamb():
    x = torch.linspace(-3, 3, 31, dtype=torch.double)
    assert torch.allclose(LogPenalty(lamb=0.2).prox(x, step=2.5), LogPenalty(lamb=0.5).prox(x, step=1.0))


def test_prox_is_gradcheck_clean():
    torch.manual_seed(1)
    sign = torch.sign(torch.rand(8, dtype=torch.double) - 0.5)
    x = (sign * (0.3 + torch.rand(8, dtype=torch.double) * 1.2)).requires_grad_(True)
    penalty = LogPenalty(lamb=0.1)

    assert torch.autograd.gradcheck(lambda t: penalty.prox(t, 1.0), (x,), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("beta", [0.5, 1.0, 4.0])
def test_value_and_grad_are_consistent(beta):
    torch.manual_seed(2)
    x = (torch.rand(16, dtype=torch.double) * 2 - 1).requires_grad_(True)
    penalty = LogPenalty(lamb=0.3, beta=beta)

    val = penalty.value(x)
    val.backward()

    analytic = penalty.grad(x.detach())
    np.testing.assert_allclose(x.grad.numpy(), analytic.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kwargs", [{"lamb": 0.0}, {"lamb": 0.1, "beta": 0.0}])
def test_invalid_params_rejected(kwargs):
    with pytest.raises(ValueError):
        LogPenalty(**kwargs)


def test_certifies_invex_only_at_beta_1():
    penalty = LogPenalty(lamb=0.1)

    assert penalty.invex is not None
    assert penalty.invex.reference.locator == "Lemma 1, item 2 (Eq. 7)"
    assert penalty.quasi_invex is penalty.invex
    assert penalty.convex is None and penalty.quasi_convex is None

    # beta != 1 has no manual invex certificate, but is invex via admissibility
    derived = LogPenalty(lamb=0.1, beta=2.0).invex
    assert derived.source == "rule:admissible_invex"
    assert derived.derived_from[0].reference.locator.startswith("Definition 4")
