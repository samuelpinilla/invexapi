import numpy as np
import pytest
import torch

from invexapi import LogMinusRationalPenalty
from tests.reference_numpy import log_minus_rational_prox_np
from tests.test_log_penalty import brute_force_prox


@pytest.mark.parametrize("beta", [1.0, 2.0, 3.5])
@pytest.mark.parametrize("lamb", [1e-3, 0.05, 0.5])
def test_prox_matches_numpy_reference(lamb, beta):
    torch.manual_seed(0)
    x = (torch.rand(64) * 4 - 2).double()

    penalty = LogMinusRationalPenalty(lamb=lamb, beta=beta)
    y_torch = penalty.prox(x, step=1.0).numpy()

    y_ref = log_minus_rational_prox_np(x.numpy(), lamb, beta)

    np.testing.assert_allclose(y_torch, y_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("lamb", [0.5, 8.0, 30.0])  # 8 and 30 exceed the convex limit 27/4
def test_prox_is_global_minimizer(lamb):
    penalty = LogMinusRationalPenalty(lamb=lamb)
    x = torch.linspace(-12, 12, 97, dtype=torch.double)

    y = penalty.prox(x, step=1.0)

    np.testing.assert_allclose(y.numpy(), brute_force_prox(penalty, x, lamb), atol=2e-3)


def test_prox_is_gradcheck_clean():
    torch.manual_seed(1)
    sign = torch.sign(torch.rand(8, dtype=torch.double) - 0.5)
    x = (sign * (0.3 + torch.rand(8, dtype=torch.double) * 1.2)).requires_grad_(True)
    penalty = LogMinusRationalPenalty(lamb=0.1)

    assert torch.autograd.gradcheck(lambda t: penalty.prox(t, 1.0), (x,), eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("beta", [1.0, 2.0, 5.0])
def test_value_and_grad_are_consistent(beta):
    torch.manual_seed(2)
    x = (torch.rand(16, dtype=torch.double) * 2 - 1).requires_grad_(True)
    penalty = LogMinusRationalPenalty(lamb=0.3, beta=beta)

    val = penalty.value(x)
    val.backward()

    analytic = penalty.grad(x.detach())
    np.testing.assert_allclose(x.grad.numpy(), analytic.numpy(), rtol=1e-5, atol=1e-6)


def test_value_is_weighted_by_lamb():
    x = torch.tensor([0.5, -2.0], dtype=torch.double)
    assert torch.isclose(LogMinusRationalPenalty(lamb=3.0).value(x), 3.0 * LogMinusRationalPenalty(lamb=1.0).value(x))


@pytest.mark.parametrize("kwargs", [{"lamb": 0.0}, {"lamb": 0.1, "beta": 0.5}])
def test_invalid_params_rejected(kwargs):
    with pytest.raises(ValueError):
        LogMinusRationalPenalty(**kwargs)


def test_certifies_invex_only_at_beta_2():
    penalty = LogMinusRationalPenalty(lamb=0.1)

    assert penalty.invex is not None
    assert penalty.invex.status == "assumed"
    assert penalty.invex.reference.locator == "Lemma 1, item 5 (Eq. 10)"
    assert penalty.quasi_invex is penalty.invex  # invex => quasi-invex, unconditionally

    assert penalty.convex is None
    assert penalty.quasi_convex is None

    derived = LogMinusRationalPenalty(lamb=0.1, beta=3.0).invex  # via admissibility only
    assert derived.source == "rule:admissible_invex"
