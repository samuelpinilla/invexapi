import warnings

import torch

from invexapi import Sum, TikhonovPenalty
from invexapi.optim import FISTA, GradientDescent, NonlinearCG


class _Uncertified:
    """Plain duck-typed objective with no Loss/certificate machinery at all."""

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(x * x)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_tikhonov_certifies_all_four_properties():
    penalty = TikhonovPenalty(lamb=0.5)

    assert penalty.convex is not None
    assert penalty.invex is not None
    assert penalty.quasi_convex is not None
    assert penalty.quasi_invex is penalty.invex  # derived fallback, not separately set


def test_sum_never_auto_derives_certificates():
    a = TikhonovPenalty(lamb=0.5)
    b = TikhonovPenalty(lamb=0.1)

    combined = Sum(a, b)

    assert combined.invex is None
    assert combined.convex is None
    assert combined.quasi_convex is None
    assert combined.quasi_invex is None


def test_gradient_descent_warns_on_uncertified_objective():
    objective = _Uncertified()
    solver = GradientDescent(objective, step=0.5, max_iter=5)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver.run(torch.ones(4))

    assert any("certificate" in str(w.message) for w in caught)


def test_nonlinear_cg_warns_on_uncertified_objective():
    objective = _Uncertified()
    solver = NonlinearCG(objective, step=0.5, max_iter=5)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver.run(torch.ones(4))

    assert any("certificate" in str(w.message) for w in caught)


def test_fista_warns_on_uncertified_combined_objective():
    smooth = _Uncertified()
    penalty = TikhonovPenalty(lamb=0.1)
    solver = FISTA(smooth, penalty, step=0.5, max_iter=5)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver.run(torch.ones(4))

    assert any("certificate" in str(w.message) for w in caught)


def test_fista_does_not_warn_once_combined_is_manually_certified():
    smooth = TikhonovPenalty(lamb=0.2)
    penalty = TikhonovPenalty(lamb=0.1)
    solver = FISTA(smooth, penalty, step=0.5, max_iter=5)

    # Both parts are convex, and convexity DOES compose additively — a human
    # verifying that fact for this specific combined objective attaches it here,
    # rather than the library ever inferring it automatically (see Sum's docstring).
    solver.combined._certify("convex", smooth.convex)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver.run(torch.ones(4))

    assert not any("certificate" in str(w.message) for w in caught)
