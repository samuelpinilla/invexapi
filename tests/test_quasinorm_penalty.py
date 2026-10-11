import numpy as np
import pytest
import torch

from invexapi import QuasinormInvexPenalty
from invexapi.functions.separable.quasinorm import _QuasinormProx
from tests.reference_numpy import quasinorm_prox_np


@pytest.mark.parametrize("lamb,q", [(0.05, 0.5), (0.1, 0.3), (0.2, 0.7)])
def test_prox_matches_numpy_reference(lamb, q):
    torch.manual_seed(0)
    x = (torch.rand(64) * 4 - 2).double()

    penalty = QuasinormInvexPenalty(lamb=lamb, q=q)
    y_torch = penalty.prox(x, step=1.0).numpy()

    y_ref = quasinorm_prox_np(x.numpy(), lamb, q)

    np.testing.assert_allclose(y_torch, y_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("lamb,q", [(0.05, 0.5), (0.1, 0.3), (0.2, 0.7)])
def test_prox_backward_is_exact_away_from_threshold(lamb, q):
    # 4 fixed-point steps are converged except right at the threshold, so the
    # implicit-function backward must match finite differences of the forward here.
    x = torch.tensor([-2.5, -1.7, -0.01, 0.02, 1.3, 2.9], dtype=torch.float64, requires_grad=True)
    lam = torch.tensor(lamb, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t, l: _QuasinormProx.apply(t, l, q), (x, lam), eps=1e-6, atol=1e-5)


def test_prox_backward_is_zero_where_prox_thresholds():
    penalty = QuasinormInvexPenalty(lamb=0.5, q=0.5)
    x = torch.tensor([0.3, -0.6, 2.0], dtype=torch.float64, requires_grad=True)
    y = penalty.prox(x, step=1.0)
    y.sum().backward()
    assert y[0] == 0 and y[1] == 0 and x.grad[0] == 0 and x.grad[1] == 0
    z = y[2].item()
    assert x.grad[2].item() == pytest.approx(1.0 / (1.0 + 0.5 * 0.5 * (0.5 - 1.0) * z ** -1.5))


def test_value_and_grad_are_consistent():
    torch.manual_seed(2)
    x = (torch.rand(16, dtype=torch.double) * 2 + 0.1).requires_grad_(True)
    penalty = QuasinormInvexPenalty(lamb=0.1, q=0.5)

    val = penalty.value(x)
    val.backward()

    analytic = penalty.grad(x.detach())
    np.testing.assert_allclose(x.grad.numpy(), analytic.numpy(), rtol=1e-5, atol=1e-6)


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        QuasinormInvexPenalty(lamb=0.1, q=1.5)
    with pytest.raises(ValueError):
        QuasinormInvexPenalty(lamb=-1.0, q=0.5)


def test_certifies_invex_only():
    penalty = QuasinormInvexPenalty(lamb=0.1, q=0.5)

    assert penalty.invex is not None
    assert penalty.invex.status == "assumed"
    assert penalty.invex.reference.locator == "Lemma 1, item 1 (Eq. 6)"
    assert penalty.quasi_invex is penalty.invex  # invex => quasi-invex, unconditionally

    assert penalty.convex is None
    assert penalty.quasi_convex is None
