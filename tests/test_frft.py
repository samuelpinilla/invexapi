import math

import pytest
import torch

from invexapi.operators import FractionalFourier, LiftedFrFTAmbiguity, frft, frft_matrix, frft_orders


def _gaussian(n: int, center: float) -> torch.Tensor:
    t = (torch.arange(n, dtype=torch.float64) - center) / math.sqrt(n)
    return torch.exp(-math.pi * t**2).to(torch.complex128)


def test_order_one_is_the_centered_unitary_dft():
    torch.manual_seed(0)
    n = 16
    x = torch.randn(n, dtype=torch.complex128)
    shft = (torch.arange(n) + n // 2) % n
    expected = torch.empty_like(x)
    expected[shft] = torch.fft.fft(x[shft]) / math.sqrt(n)
    assert torch.allclose(frft(x, 1.0), expected, atol=1e-12)


def test_trivial_orders():
    x = torch.randn(9, dtype=torch.complex128)
    assert torch.allclose(frft(x, 0.0), x)
    assert torch.allclose(frft(x, 4.0), x)
    assert torch.allclose(frft(x, 2.0), torch.flip(x, dims=(-1,)))


@pytest.mark.parametrize("a", [0.5, 0.73, 1.21, 1.5, -0.7, 2.6, 3.4])
def test_gaussian_is_an_eigenfunction_for_odd_length(a):
    # exp(-pi t^2) is invariant under every FrFT order; the fast algorithm
    # reproduces that to machine precision on its own sampling grid.
    g = _gaussian(127, 63)
    assert (frft(g, a) - g).norm() / g.norm() < 1e-10


def test_even_length_general_orders_are_centered_at_half_sample():
    # the general-order chirp grid is centered at (N-1)/2 for even N - kept on purpose.
    g = _gaussian(128, 63.5)
    assert (frft(g, 0.8) - g).norm() / g.norm() < 1e-10


def test_batched_rows_match_single_calls():
    torch.manual_seed(1)
    X = torch.randn(3, 20, dtype=torch.complex128)
    out = frft(X, 0.9)
    for row in range(3):
        assert torch.allclose(out[row], frft(X[row], 0.9))


def test_matrix_and_operator_agree_with_function():
    torch.manual_seed(2)
    x = torch.randn(24, dtype=torch.complex128)
    op = FractionalFourier(24, 1.3)
    assert torch.allclose(frft_matrix(24, 1.3) @ x, frft(x, 1.3), atol=1e-12)
    assert torch.allclose(op.apply(x), frft(x, 1.3), atol=1e-12)


def _adjoint_gap(op, x, y):
    lhs = torch.vdot(op.apply(x).reshape(-1), y.reshape(-1))
    rhs = torch.vdot(x.reshape(-1), op.adjoint(y).reshape(-1))
    return (lhs - rhs).abs().item() / max(lhs.abs().item(), 1.0)


def test_fractional_fourier_adjoint_identity():
    torch.manual_seed(3)
    op = FractionalFourier(32, 0.6)
    x = torch.randn(32, dtype=torch.complex128)
    y = torch.randn(32, dtype=torch.complex128)
    assert _adjoint_gap(op, x, y) < 1e-12


def test_frft_orders_are_inclusive_linspace():
    orders = frft_orders(128, 3)
    assert len(orders) == 43
    assert orders[0] == 0.5 and orders[-1] == 1.5


def test_lifted_ambiguity_adjoint_identity():
    torch.manual_seed(4)
    n = 16
    op = LiftedFrFTAmbiguity(n, frft_orders(n, 3))
    X = torch.randn(n, n, dtype=torch.complex128)
    y = torch.randn(*op.shape, dtype=torch.complex128)
    assert _adjoint_gap(op, X, y) < 1e-12


def test_lifted_ambiguity_of_rank_one_is_fft_of_frft_intensity():
    torch.manual_seed(5)
    n = 20
    op = LiftedFrFTAmbiguity(n, frft_orders(n, 2))
    x = torch.randn(n, dtype=torch.complex128)
    lifted = op.apply(torch.outer(x, x.conj()))
    for j, a in enumerate(op.orders):
        expected = torch.fft.fft(frft(x, float(a)).abs() ** 2)
        assert torch.allclose(lifted[j], expected, atol=1e-10)
    assert torch.allclose(op.measure(x), lifted, atol=1e-10)
    assert torch.allclose(op.ambiguity(x), lifted.abs() ** 2, atol=1e-8)


def test_lifted_ambiguity_matches_explicit_trace_form():
    # explicit B_{alpha,k} = F_a^H diag(DFT row k) F_a and Tr(B X), as in Eq. (13).
    torch.manual_seed(6)
    n = 8
    op = LiftedFrFTAmbiguity(n, [0.7, 1.2])
    X = torch.randn(n, n, dtype=torch.complex128)
    dft = torch.fft.fft(torch.eye(n, dtype=torch.complex128), dim=0)
    for j, F in enumerate(op.matrices):
        for k in range(n):
            B = F.conj().T @ torch.diag(dft[k]) @ F
            assert torch.allclose(op.apply(X)[j, k], torch.trace(B @ X), atol=1e-10)


def test_signal_operator_matches_lifted_ambiguity_and_is_differentiable():
    from torch.func import jacfwd

    from invexapi.operators import FrFTAmbiguity

    torch.manual_seed(7)
    n = 12
    lifted = LiftedFrFTAmbiguity(n, frft_orders(n, 2))
    op = lifted.signal_operator()
    assert isinstance(op, FrFTAmbiguity) and op.matrices is lifted.matrices
    x = torch.randn(n, dtype=torch.complex128)
    assert torch.allclose(op(x), lifted.ambiguity(x), atol=1e-10)

    v = torch.view_as_real(x).reshape(-1)
    J = jacfwd(lambda w: op(torch.view_as_complex(w.reshape(n, 2))).reshape(-1))(v)
    d = torch.randn_like(v) * 1e-6
    lin = (op(torch.view_as_complex((v + d).reshape(n, 2))) - op(x)).reshape(-1)
    assert torch.allclose(lin, J @ d, rtol=1e-4, atol=1e-12)


@pytest.mark.parametrize("pad", [2, 3])
def test_padded_lifted_ambiguity_is_the_frft_of_the_centred_zero_padded_signal(pad):
    torch.manual_seed(8)
    n = 11
    op = LiftedFrFTAmbiguity(n, frft_orders(n, 2), pad=pad)
    m = pad * n
    assert op.shape == (len(op.orders), m) and op.matrices.shape == (len(op.orders), m, n)
    x = torch.randn(n, dtype=torch.complex128)
    lo = m // 2 - n // 2
    xp = torch.zeros(m, dtype=torch.complex128)
    xp[lo:lo + n] = x
    for j, a in enumerate(op.orders):
        assert torch.allclose(op.measure(x)[j], torch.fft.fft(frft(xp, float(a)).abs() ** 2), atol=1e-10)
    X = torch.randn(n, n, dtype=torch.complex128)
    y = torch.randn(*op.shape, dtype=torch.complex128)
    assert _adjoint_gap(op, X, y) < 1e-12
    assert torch.allclose(op.signal_operator()(x), op.ambiguity(x), atol=1e-9)


def test_padding_makes_time_shift_a_much_more_exact_ambiguity():
    # a band-limited sub-sample shift leaves the continuous FrFT-based AF unchanged;
    # the fast FrFT only approximates that, much better on a padded grid
    n = 32
    t = torch.arange(n, dtype=torch.float64) - n / 2
    x = (torch.exp(-((t - 4) / 4) ** 2) * torch.exp(1j * 0.08 * t**2)).to(torch.complex128)
    w = torch.fft.fftfreq(n, dtype=torch.float64) * 2 * math.pi
    xs = torch.fft.ifft(torch.fft.fft(x) * torch.exp(-1j * w * 0.3))
    gaps = []
    for pad in (1, 2):
        op = LiftedFrFTAmbiguity(n, frft_orders(n, 2), pad=pad)
        A = op.ambiguity(x)
        gaps.append(((op.ambiguity(xs) - A).norm() / A.norm()).item())
    assert gaps[1] < gaps[0] / 10
