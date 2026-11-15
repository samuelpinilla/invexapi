import numpy as np
import pytest
import torch

from invexapi.operators import (
    LinearOperator, LiftedRadarAmbiguity, NonlinearOperator, RadarAmbiguity, check_adjoint, integer_grid,
)


def _signal(n=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dtype=torch.complex128, generator=g)


def _eq2(x: np.ndarray) -> np.ndarray:
    n = np.arange(len(x))
    return np.array([[np.sum(x * np.conj(x[(n - p) % len(x)]) * np.exp(-2j * np.pi * n * k / len(x)))
                      for k in range(len(x))] for p in range(len(x))])


def test_class_hierarchy():
    assert issubclass(LiftedRadarAmbiguity, LinearOperator)
    assert issubclass(RadarAmbiguity, NonlinearOperator) and not issubclass(RadarAmbiguity, LinearOperator)


def test_integer_grid_reproduces_eq2():
    x = _signal()
    op = LiftedRadarAmbiguity(len(x))
    ref = _eq2(x.numpy())
    assert np.allclose(op.apply(torch.outer(x, x.conj())).numpy(), ref, atol=1e-10)
    assert np.allclose(op.measure(x).numpy(), ref, atol=1e-10)
    assert np.allclose(op.signal_operator()(x).numpy(), np.abs(ref) ** 2, atol=1e-10)


@pytest.mark.parametrize("delays,dopplers", [(None, None), ([0.0, 1.5, -3.25], [0.25, 4.0, 9.5, 17.0])])
def test_adjoint_and_measure_consistency(delays, dopplers):
    x = _signal()
    op = LiftedRadarAmbiguity(len(x), delays, dopplers)
    check_adjoint(op, x=torch.randn(len(x), len(x), dtype=torch.complex128))
    assert torch.allclose(op.apply(torch.outer(x, x.conj())), op.measure(x), atol=1e-12)
    assert op.shape == op.apply(torch.eye(len(x))).shape


def test_full_grid_gram_is_n_identity():
    op = LiftedRadarAmbiguity(16)
    X = torch.randn(16, 16, dtype=torch.complex128)
    assert torch.allclose(op.adjoint(op.apply(X)), 16 * X, atol=1e-10)


def test_integer_grid_decimation():
    assert integer_grid(10, 3).tolist() == [0.0, 3.0, 6.0, 9.0]
    with pytest.raises(ValueError):
        integer_grid(10, 0)
