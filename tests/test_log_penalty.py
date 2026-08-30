import numpy as np
import pytest
import torch

from invexapi import LogInvexPenalty
from tests.reference_numpy import log_prox_np


@pytest.mark.parametrize("lamb", [1e-3, 0.05, 0.5])
def test_prox_matches_reference_kernel(lamb):
    torch.manual_seed(0)
    x = (torch.rand(64) * 4 - 2).double()

    penalty = LogInvexPenalty(lamb=lamb)
    y_torch = penalty.prox(x, step=1.0).numpy()

    y_ref = log_prox_np(x.numpy(), lamb)

    np.testing.assert_allclose(y_torch, y_ref, rtol=1e-4, atol=1e-5)


def test_prox_is_gradcheck_clean():
    torch.manual_seed(1)
    sign = torch.sign(torch.rand(8, dtype=torch.double) - 0.5)
    x = (sign * (0.3 + torch.rand(8, dtype=torch.double) * 1.2)).requires_grad_(True)
    penalty = LogInvexPenalty(lamb=0.1)

    assert torch.autograd.gradcheck(lambda t: penalty.prox(t, 1.0), (x,), eps=1e-6, atol=1e-4)


def test_value_and_grad_are_consistent():
    torch.manual_seed(2)
    x = (torch.rand(16, dtype=torch.double) * 2 - 1).requires_grad_(True)
    penalty = LogInvexPenalty(lamb=0.1)

    val = penalty.value(x)
    val.backward()

    analytic = penalty.grad(x.detach())
    np.testing.assert_allclose(x.grad.numpy(), analytic.numpy(), rtol=1e-5, atol=1e-6)


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        LogInvexPenalty(lamb=0.0)


def test_certifies_invex_only():
    penalty = LogInvexPenalty(lamb=0.1)

    assert penalty.invex is not None
    assert penalty.invex.status == "assumed"
    assert penalty.invex.reference.locator == "Lemma 1, item 5 (Eq. 10)"
    assert penalty.quasi_invex is penalty.invex  # invex => quasi-invex, unconditionally

    assert penalty.convex is None
    assert penalty.quasi_convex is None
