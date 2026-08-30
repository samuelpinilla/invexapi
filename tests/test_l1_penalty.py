import torch

from invexapi import L1Penalty


def test_prox_matches_soft_threshold_closed_form():
    penalty = L1Penalty(lamb=0.5)
    x = torch.tensor([-2.0, -0.3, 0.0, 0.3, 2.0])

    got = penalty.prox(x, step=1.0)
    expected = torch.sign(x) * torch.clamp(x.abs() - 0.5, min=0.0)

    assert torch.allclose(got, expected)


def test_certifies_convex_invex_quasi_convex():
    penalty = L1Penalty(lamb=0.1)

    assert penalty.convex is not None
    assert penalty.invex is not None
    assert penalty.quasi_convex is not None
    assert penalty.quasi_invex is penalty.invex


def test_value_and_grad_are_consistent_away_from_kink():
    x = torch.tensor([1.0, 2.0, -3.0], requires_grad=True)
    penalty = L1Penalty(lamb=0.2)

    penalty.value(x).backward()
    assert torch.allclose(x.grad, penalty.grad(x.detach()))
