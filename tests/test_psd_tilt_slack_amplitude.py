import torch

from invexapi import SlackAmplitudeLeastSquares, LinearTilt, PSDTracePenalty


def _random_matrix(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, n, dtype=torch.complex128, generator=g)


def test_psd_prox_is_the_hermitian_psd_projection_for_zero_trace_weight():
    X = _random_matrix(6, 0)
    P = PSDTracePenalty(lamb=0.0).prox(X, step=1.0)

    assert torch.allclose(P, P.conj().T, atol=1e-12)
    assert torch.linalg.eigvalsh(P).min() > -1e-10
    # projection is idempotent and its residual is orthogonal to the cone point
    assert torch.allclose(PSDTracePenalty(0.0).prox(P, 1.0), P, atol=1e-10)
    H = 0.5 * (X + X.conj().T)
    assert abs(torch.vdot((H - P).reshape(-1), P.reshape(-1)).real) < 1e-10


def test_psd_prox_soft_thresholds_eigenvalues():
    d = torch.tensor([3.0, 1.0, -2.0], dtype=torch.float64)
    Q, _ = torch.linalg.qr(_random_matrix(3, 1))
    X = Q @ torch.diag(d).to(Q.dtype) @ Q.conj().T
    P = PSDTracePenalty(lamb=0.5).prox(X, step=2.0)
    got = torch.sort(torch.linalg.eigvalsh(P)).values
    assert torch.allclose(got, torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64), atol=1e-10)


def test_psd_value_is_trace_inside_cone_and_inf_outside():
    penalty = PSDTracePenalty(lamb=2.0)
    assert penalty.value(torch.eye(3, dtype=torch.complex128)).item() == 6.0
    assert penalty.value(-torch.eye(3, dtype=torch.complex128)).item() == float("inf")


def test_psd_certifies_convex_and_tilt_keeps_only_convexity():
    base = PSDTracePenalty(lamb=0.1)
    tilted = LinearTilt(base, torch.eye(2, dtype=torch.complex128))
    assert base.convex is not None and base.invex is not None
    assert tilted.convex.source == "rule:convex_tilt"
    assert tilted.invex.source == "rule:convex_invex"
    assert tilted.quasi_convex.source == "rule:convex_quasi_convex"  # via its convexity


def test_tilt_prox_minimizes_tilted_objective():
    # prox_{t g}(x) = argmin_z t*(base(z) - Re<c,z>) + 0.5||z-x||^2; for the PSD
    # base, compare against perturbations that stay inside the cone.
    base = PSDTracePenalty(lamb=0.3)
    c = _random_matrix(4, 2)
    c = 0.5 * (c + c.conj().T)
    tilt = LinearTilt(base, c)
    x = _random_matrix(4, 3)
    t = 0.7
    z = tilt.prox(x, t)

    def obj(w):
        return t * tilt.value(w) + 0.5 * (w - x).abs().pow(2).sum()

    best = obj(z)
    for seed in range(20):
        w = base.prox(z + 1e-3 * _random_matrix(4, 100 + seed), 0.0)
        assert obj(w) >= best - 1e-12


def test_slack_amplitude_least_squares_value_and_grad():
    b = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    s = torch.tensor([0.0, 0.5, 0.0], dtype=torch.float64)
    loss = SlackAmplitudeLeastSquares(b, slack=s)
    z = torch.tensor([1 + 1j, -0.5j, 2.0], dtype=torch.complex128)
    r = z.abs() + s - b
    assert torch.isclose(loss.value(z), 0.5 * (r**2).sum())
    assert torch.allclose(loss.grad(z), z / z.abs() * r)
    assert loss.convex is None and loss.invex is None
