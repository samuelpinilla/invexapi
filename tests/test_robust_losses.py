import math

import pytest
import torch

from invexapi import (
    AdaptiveRobustLoss,
    CauchyLoss,
    GemanMcClureLoss,
    SquaredLogMinusRationalLoss,
    WelschLoss,
    catalog,
)
from invexapi.core import NumericalVerifier
from invexapi.functions import Composed
from invexapi.operators import MatrixOperator, Subsample

_ADMISSIBLE = [
    CauchyLoss(0.7),
    GemanMcClureLoss(1.3),
    WelschLoss(0.5),
    SquaredLogMinusRationalLoss(),
    AdaptiveRobustLoss(-math.inf, 0.8),
    AdaptiveRobustLoss(-3.0, 0.8),
    AdaptiveRobustLoss(0.0, 0.8),
    AdaptiveRobustLoss(1.0, 0.8),
    AdaptiveRobustLoss(2.0, 0.8),
]


@pytest.mark.parametrize("loss", _ADMISSIBLE + [AdaptiveRobustLoss(3.0, 0.8)], ids=lambda l: f"{type(l).__name__}{getattr(l, 'alpha', '')}")
def test_grad_matches_autograd(loss):
    x = torch.linspace(-4, 4, 41, dtype=torch.float64, requires_grad=True)
    loss.value(x).backward()
    assert torch.allclose(x.grad, loss.grad(x.detach()), atol=1e-12)


@pytest.mark.parametrize("loss", _ADMISSIBLE, ids=lambda l: f"{type(l).__name__}{getattr(l, 'alpha', '')}")
def test_admissible_certificate_cites_theorem_5_and_passes_def_4(loss):
    assert loss.admissible.reference.locator.startswith("Theorem 5, Eq.")
    assert loss.invex.source == "rule:admissible_invex"
    assert loss.differentiable is not None
    assert NumericalVerifier().check(loss, "admissible") is not None


def test_adaptive_alpha_above_two_is_not_admissible():
    loss = AdaptiveRobustLoss(3.0)
    assert loss.admissible is None and loss.invex is None
    assert loss.differentiable is not None
    assert NumericalVerifier().check(loss, "admissible") is None


@pytest.mark.parametrize("limit, near", [(2.0, 2.0 + 1e-7), (0.0, 1e-7), (-math.inf, -1e7)])
def test_adaptive_limits_are_continuous(limit, near):
    x = torch.linspace(-3, 3, 13, dtype=torch.float64)
    assert torch.allclose(AdaptiveRobustLoss(limit).value(x), AdaptiveRobustLoss(near).value(x), rtol=1e-4)


def test_adaptive_special_cases_match_the_named_losses():
    x = torch.linspace(-3, 3, 13, dtype=torch.float64)
    c = 0.9
    assert torch.allclose(AdaptiveRobustLoss(-2.0, c).value(x), GemanMcClureLoss(c).value(x))
    assert torch.allclose(AdaptiveRobustLoss(-math.inf, c).value(x), WelschLoss(c).value(x))
    # Eq. (4)'s Cauchy uses delta^2 = 2 c^2 relative to the alpha = 0 limit.
    assert torch.allclose(AdaptiveRobustLoss(0.0, c).value(x), CauchyLoss(math.sqrt(2) * c).value(x))


def test_rejects_bad_parameters():
    for make in (lambda: CauchyLoss(0.0), lambda: WelschLoss(-1.0), lambda: AdaptiveRobustLoss(1.0, c=0.0),
                 lambda: AdaptiveRobustLoss(math.nan), lambda: AdaptiveRobustLoss(math.inf)):
        with pytest.raises(ValueError):
            make()


def test_robust_regression_is_certified_invex_with_full_row_rank_operator():
    y = torch.randn(3, dtype=torch.float64)
    H = Subsample(torch.tensor([True, True, False, True]))  # full row rank
    objective = Composed(CauchyLoss(1.0), H, b=y)
    assert objective.invex.source == "rule:affine_invex_construction"
    unknown_rank = Composed(CauchyLoss(1.0), MatrixOperator(torch.randn(3, 4, dtype=torch.float64)), b=y)
    assert unknown_rank.invex is None


def test_catalog_lists_the_losses():
    assert {CauchyLoss, GemanMcClureLoss, WelschLoss, AdaptiveRobustLoss, SquaredLogMinusRationalLoss} <= set(catalog(invex=True, convex=False))
