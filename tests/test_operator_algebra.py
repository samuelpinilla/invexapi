import pytest
import torch

from invexapi.operators import (
    Convolution,
    Diagonal,
    FiniteDifference2D,
    FractionalFourier,
    FrFTAmbiguity,
    Identity,
    LiftedFrFTAmbiguity,
    LinearOperator,
    MatrixOperator,
    NonlinearOperator,
    Subsample,
    check_adjoint,
    frft_orders,
)


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _real_matrix(m, n, seed=0):
    return torch.randn(m, n, generator=_gen(seed), dtype=torch.float64)


def _complex_matrix(m, n, seed=0):
    g = _gen(seed)
    return torch.complex(torch.randn(m, n, generator=g, dtype=torch.float64), torch.randn(m, n, generator=g, dtype=torch.float64))


def _linear_ops():
    mask = torch.rand(5, 6, generator=_gen(3)) > 0.4
    kernel = torch.randn(3, 3, generator=_gen(4), dtype=torch.float64)
    A = MatrixOperator(_real_matrix(4, 6))
    B = MatrixOperator(_real_matrix(6, 6, seed=1))
    C = MatrixOperator(_complex_matrix(3, 6, seed=2))
    return {
        "identity": Identity((4, 3)),
        "matrix": A,
        "complex_matrix": C,
        "diagonal": Diagonal(torch.randn(7, generator=_gen(5), dtype=torch.float64)),
        "convolution": Convolution(kernel, (8, 9)),
        "complex_convolution": Convolution(kernel.to(torch.complex128) * (1 + 2j), (8, 9)),
        "subsample": Subsample(mask),
        "finite_difference": FiniteDifference2D((6, 5)),
        "frft": FractionalFourier(8, 0.7),
        "lifted_ambiguity": LiftedFrFTAmbiguity(6, frft_orders(6), pad=2),
        "composition": A @ B,
        "sum": A + 2.0 * A,
        "difference": B - B.H,
        "complex_scale": (1 - 3j) * C,
        "adjoint": C.H,
        "subsample_after_convolution": Subsample(torch.rand(8, 9, generator=_gen(6)) > 0.5) @ Convolution(kernel, (8, 9)),
    }


@pytest.mark.parametrize("name", list(_linear_ops()))
def test_adjoints_pass_inner_product_identity(name):
    op = _linear_ops()[name]
    assert isinstance(op, LinearOperator)
    if name == "frft" or name == "lifted_ambiguity":
        x = torch.randn(op.in_shape, generator=_gen(7), dtype=torch.float64).to(torch.complex128)
        check_adjoint(op, x=x)
    else:
        check_adjoint(op)


def test_check_adjoint_catches_a_wrong_adjoint():
    class Wrong(MatrixOperator):
        def adjoint(self, v):
            return v @ self.matrix.T  # applies M again instead of M^T

    with pytest.raises(AssertionError, match="adjoint mismatch"):
        check_adjoint(Wrong(_real_matrix(4, 4)))


def test_norm_matches_spectral_norm():
    M = _complex_matrix(5, 7)
    expected = torch.linalg.matrix_norm(M, ord=2).item()
    assert MatrixOperator(M).norm(like=torch.zeros(7, dtype=torch.complex128)) == pytest.approx(expected, rel=1e-5)
    assert (3.0 * MatrixOperator(M)).norm(like=torch.zeros(7, dtype=torch.complex128)) == pytest.approx(3 * expected, rel=1e-5)


def test_norm_of_convolution_is_peak_transfer_gain():
    conv = Convolution(torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float64), (16, 16))
    assert conv.norm() == pytest.approx(conv.transfer.abs().max().item(), rel=1e-4)


def test_norm_needs_a_shape():
    with pytest.raises(ValueError):
        Identity().norm()


def test_matmul_with_tensor_applies():
    A = MatrixOperator(_real_matrix(3, 4))
    x = torch.randn(4, dtype=torch.float64)
    assert torch.allclose(A @ x, A.matrix @ x)


def test_centred_delta_convolution_is_identity():
    delta = torch.zeros(3, 5, dtype=torch.float64)
    delta[1, 2] = 1.0
    x = torch.randn(7, 9, dtype=torch.float64)
    assert torch.allclose(Convolution(delta, (7, 9))(x), x)


def test_convolution_matches_direct_circular_sum():
    kernel = torch.randn(3, generator=_gen(1), dtype=torch.float64)
    x = torch.randn(10, generator=_gen(2), dtype=torch.float64)
    direct = sum(kernel[j] * torch.roll(x, j - 1) for j in range(3))
    assert torch.allclose(Convolution(kernel, (10,))(x), direct)


def test_composition_propagates_only_provable_properties():
    surjective = MatrixOperator(_real_matrix(3, 5)).declare(full_row_rank=True)
    other = MatrixOperator(_real_matrix(5, 5, seed=1)).declare(full_row_rank=True, injective=True)
    unknown = MatrixOperator(_real_matrix(5, 5, seed=2))

    assert (surjective @ other).full_row_rank is True
    assert (surjective @ unknown).full_row_rank is None  # unknown, never False
    assert (surjective @ other).injective is None
    assert (Diagonal(torch.ones(4)) @ Identity()).diagonal is True


def test_adjoint_swaps_rank_and_injectivity():
    op = MatrixOperator(_real_matrix(3, 5)).declare(full_row_rank=True, injective=False)
    assert op.H.injective is True and op.H.full_row_rank is False
    assert op.H.in_shape == (3,) and op.H.out_shape == (5,)
    assert op.H.H is op


def test_sum_and_scaling_rules():
    d = Diagonal(torch.tensor([1.0, 2.0]))
    assert (d + d).diagonal is True
    assert (d + d).full_row_rank is None  # a sum can lose rank
    assert (2.0 * d).full_row_rank is True
    assert (0.0 * d).full_row_rank is None
    assert (-1.0 * Identity()).unitary is True and (2.0 * Identity()).unitary is None


def test_structural_properties_are_exact_where_set():
    assert Diagonal(torch.tensor([1.0, 0.0])).full_row_rank is False
    assert Diagonal(torch.tensor([1.0, 3.0])).injective is True
    assert Subsample(torch.tensor([True, False])).full_row_rank is True
    assert Subsample(torch.tensor([True, False])).injective is False
    assert FiniteDifference2D().full_row_rank is False and FiniteDifference2D().injective is False
    assert MatrixOperator(_real_matrix(2, 2)).full_row_rank is None


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        MatrixOperator(_real_matrix(3, 4)) @ MatrixOperator(_real_matrix(5, 6))
    with pytest.raises(ValueError, match="shape mismatch"):
        MatrixOperator(_real_matrix(3, 4)) + MatrixOperator(_real_matrix(3, 5))


def test_composition_with_nonlinear_is_nonlinear_and_differentiable():
    amb = FrFTAmbiguity(6, frft_orders(6), pad=2)
    lin = MatrixOperator(_real_matrix(6, 4).to(torch.complex128))
    composed = amb @ lin
    assert isinstance(composed, NonlinearOperator) and not isinstance(composed, LinearOperator)
    assert composed.in_shape == (4,) and composed.out_shape == amb.out_shape

    x = torch.randn(4, dtype=torch.float64)
    jac = torch.func.jacfwd(lambda t: composed(t.to(torch.complex128)).real)(x)
    assert jac.shape == amb.out_shape + (4,)


def test_declare_validates():
    with pytest.raises(ValueError):
        Identity().declare(convex=True)
    with pytest.raises(TypeError):
        Identity().declare(full_row_rank="yes")
