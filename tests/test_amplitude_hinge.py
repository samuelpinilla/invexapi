import warnings

import pytest
import torch

from invexapi import AmplitudeHingeSquared, LinearTilt, PSDTracePenalty, catalog
from invexapi.functions import Composed, Sum
from invexapi.operators import LiftedFrFTAmbiguity, frft_orders
from invexapi.optim import FASTA


def _complex(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dtype=torch.complex128, generator=g)


def test_value_and_grad_on_both_sides_of_the_bound():
    b = torch.tensor([1.0, 2.0, 0.5, 0.0], dtype=torch.float64)
    z = torch.tensor([1 + 1j, -0.5j, 2.0, 3j], dtype=torch.complex128)
    loss = AmplitudeHingeSquared(b)
    e = torch.clamp(z.abs() - b, min=0.0)
    assert torch.isclose(loss.value(z), 0.5 * (e**2).sum())
    assert torch.allclose(loss.grad(z), z / z.abs() * e)
    assert loss.grad(z)[1] == 0  # |z| < b: inactive, no pull


def test_grad_matches_autograd_and_is_z_minus_projection():
    b = torch.rand(20, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    loss = AmplitudeHingeSquared(b)
    z = _complex(20, 1)
    w = torch.view_as_real(z).clone().requires_grad_(True)
    loss.value(torch.view_as_complex(w)).backward()
    assert torch.allclose(torch.view_as_complex(w.grad), loss.grad(z))
    projection = z * torch.clamp(b / z.abs(), max=1.0)
    assert torch.allclose(loss.grad(z), z - projection)


def test_midpoint_convexity_spot_check():
    b = 0.5 + torch.rand(30, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    loss = AmplitudeHingeSquared(b)
    for seed in range(50):
        u, v = 2 * _complex(30, 10 + seed), 2 * _complex(30, 100 + seed)
        assert loss.value(0.5 * (u + v)) <= 0.5 * (loss.value(u) + loss.value(v)) + 1e-12


def test_certified_convex_and_c1_and_listed_in_catalog():
    loss = AmplitudeHingeSquared(torch.ones(3))
    assert loss.convex.source == "manual" and loss.differentiable is not None
    assert loss.invex.source == "rule:convex_invex"
    assert AmplitudeHingeSquared in catalog(convex=True)


def test_rejects_negative_bounds():
    with pytest.raises(ValueError):
        AmplitudeHingeSquared(torch.tensor([1.0, -0.1]))


def test_lifted_wavemax_objective_is_certified_and_fasta_stays_silent():
    n = 8
    op = LiftedFrFTAmbiguity(n, frft_orders(n, 2))
    x = _complex(n, 3)
    b = op.measure(x).abs()
    X0 = torch.outer(x, x.conj())
    smooth, penalty = AmplitudeHingeSquared(b), LinearTilt(PSDTracePenalty(0.0), 1e-2 * X0)
    assert Sum(Composed(smooth, op), penalty).convex.source == "rule:convex_sum"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FASTA(smooth, penalty, A=op, max_iter=5).run(X0)
