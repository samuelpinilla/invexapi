import numpy as np
import pytest
import torch

from invexapi import SymmetricNuclearPenalty
from invexapi.operators import FROG, LinearOperator, LiftedFROG, NonlinearOperator, check_adjoint, integer_grid


def _signal(n=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dtype=torch.complex128, generator=g)


def _eq1(x: np.ndarray, decimation: int = 1) -> np.ndarray:
    n = np.arange(len(x))
    return np.array([[np.sum(x * x[(n + p) % len(x)] * np.exp(-2j * np.pi * n * k / len(x)))
                      for k in range(len(x))] for p in range(0, len(x), decimation)])


def test_class_hierarchy():
    assert issubclass(LiftedFROG, LinearOperator)
    assert issubclass(FROG, NonlinearOperator) and not issubclass(FROG, LinearOperator)


@pytest.mark.parametrize("decimation", [1, 3])
def test_integer_grid_reproduces_eq1(decimation):
    x = _signal()
    op = LiftedFROG(len(x), integer_grid(len(x), decimation))
    ref = _eq1(x.numpy(), decimation)
    assert np.allclose(op.apply(torch.outer(x, x)).numpy(), ref, atol=1e-10)
    assert np.allclose(op.measure(x).numpy(), ref, atol=1e-10)
    assert np.allclose(op.signal_operator()(x).numpy(), np.abs(ref) ** 2, atol=1e-10)


@pytest.mark.parametrize("delays,frequencies", [(None, None), ([0.0, 1.5, -3.25], [0.25, 4.0, 9.5, 17.0])])
def test_adjoint_and_measure_consistency(delays, frequencies):
    x = _signal()
    op = LiftedFROG(len(x), delays, frequencies)
    check_adjoint(op, x=torch.randn(len(x), len(x), dtype=torch.complex128))
    assert torch.allclose(op.apply(torch.outer(x, x)), op.measure(x), atol=1e-12)


def test_full_grid_gram_is_n_identity():
    op = LiftedFROG(16)
    X = torch.randn(16, 16, dtype=torch.complex128)
    assert torch.allclose(op.adjoint(op.apply(X)), 16 * X, atol=1e-10)


def test_symmetric_nuclear_prox():
    g = torch.Generator().manual_seed(1)
    V = torch.randn(12, 12, dtype=torch.complex128, generator=g)
    P = SymmetricNuclearPenalty(lamb=0.5).prox(V, step=2.0)
    assert torch.allclose(P, P.T, atol=1e-12)
    S = 0.5 * (V + V.T)
    s_in, s_out = torch.linalg.svdvals(S), torch.linalg.svdvals(P)
    assert torch.allclose(s_out, torch.clamp(s_in - 1.0, min=0.0), atol=1e-10)
    # rank-one symmetric input: pure shrinkage
    x = _signal(12)
    X = torch.outer(x, x)
    nrm2 = torch.linalg.vector_norm(x).item() ** 2
    assert torch.allclose(SymmetricNuclearPenalty(0.3).prox(X, 1.0), (1 - 0.3 / nrm2) * X, atol=1e-10)
    assert SymmetricNuclearPenalty(1.0).value(V).isinf()
    assert torch.isclose(SymmetricNuclearPenalty(2.0).value(X), torch.tensor(2.0 * nrm2, dtype=torch.float64))
    assert SymmetricNuclearPenalty(1.0).convex
