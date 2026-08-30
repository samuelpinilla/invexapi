import torch

from invexapi.penalties.operators import FiniteDifference2D, Identity


def test_identity_is_a_no_op():
    op = Identity()
    x = torch.randn(5, 3)
    assert torch.equal(op.apply(x), x)
    assert torch.equal(op.adjoint(x), x)


def test_finite_difference_adjoint_matches_inner_product_identity():
    torch.manual_seed(0)
    op = FiniteDifference2D()
    x = torch.randn(11, 7, dtype=torch.double)
    v = torch.randn(2, 11, 7, dtype=torch.double)

    lhs = torch.sum(op.apply(x) * v)
    rhs = torch.sum(x * op.adjoint(v))

    assert torch.allclose(lhs, rhs, atol=1e-10)


def test_finite_difference_zero_on_constant_image():
    op = FiniteDifference2D()
    x = torch.full((6, 6), 3.0)
    assert torch.allclose(op.apply(x), torch.zeros(2, 6, 6))
