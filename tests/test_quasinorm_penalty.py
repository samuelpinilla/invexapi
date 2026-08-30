import numpy as np
import pytest
import torch

from invexapi import QuasinormInvexPenalty
from tests.reference_numpy import quasinorm_grad_np, quasinorm_prox_np


@pytest.mark.parametrize("lamb,q", [(0.05, 0.5), (0.1, 0.3), (0.2, 0.7)])
def test_prox_matches_reference_kernel(lamb, q):
    torch.manual_seed(0)
    x = (torch.rand(64) * 4 - 2).double()

    penalty = QuasinormInvexPenalty(lamb=lamb, q=q)
    y_torch = penalty.prox(x, step=1.0).numpy()

    y_ref = quasinorm_prox_np(x.numpy(), lamb, q)

    np.testing.assert_allclose(y_torch, y_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("lamb,q", [(0.05, 0.5), (0.1, 0.3)])
def test_prox_backward_matches_reference_gradient(lamb, q):
    torch.manual_seed(1)
    x = (torch.rand(32) * 4 - 2).double().requires_grad_(True)
    grad_y = torch.rand(32).double()

    penalty = QuasinormInvexPenalty(lamb=lamb, q=q)
    y = penalty.prox(x, step=1.0)
    y.backward(grad_y)

    _, grad_x_ref = quasinorm_grad_np(
        x.detach().numpy(), grad_y.numpy(), lamb, q
    )

    np.testing.assert_allclose(x.grad.numpy(), grad_x_ref, rtol=1e-4, atol=1e-5)


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
