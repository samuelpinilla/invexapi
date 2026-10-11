import warnings

import pytest
import torch

from invexapi import rules
from invexapi.certificate import Certificate
from invexapi.core import Loss, NumericalVerifier
from invexapi.functions import (
    Composed,
    L1Penalty,
    LogMinusRationalPenalty,
    LogPenalty,
    Max,
    QuasinormInvexPenalty,
    Sum,
    TikhonovPenalty,
)
from invexapi.metadata import Reference
from invexapi.nn import ixLinear, ixSequential
from invexapi.operators import FiniteDifference1D, FiniteDifference2D, MatrixOperator, Subsample
from invexapi.optim import FASTA, FISTA, LinearizedADMM

_REF = Reference(authors="A. Author", title="A Theorem", venue="Some Journal", year=2020, locator="Thm 1")


class _Quadratic(Loss):
    """0.5*||x||^2 with whatever certificates a test attaches."""

    def __init__(self, **certs):
        super().__init__()
        for name, cert in certs.items():
            self._certify(name, cert)

    def value(self, x):
        return 0.5 * (x * x).sum()

    def grad(self, x):
        return x


def _cert(status="assumed"):
    return Certificate(status=status, source="manual", reference=_REF)


# --- single-function implications ---------------------------------------------


def test_convex_implies_invex_and_quasi_convex():
    f = _Quadratic(convex=_cert())
    assert f.invex.source == "rule:convex_invex"
    assert f.quasi_convex.source == "rule:convex_quasi_convex"
    assert f.invex.derived_from == (f.convex,)


def test_admissible_implies_invex():
    f = _Quadratic(admissible=_cert())
    assert f.invex.source == "rule:admissible_invex"
    assert f.quasi_invex == f.invex  # invex doubles as quasi-invex (rebuilt on access, so ==)


def test_uncertified_function_derives_nothing():
    f = _Quadratic()
    assert f.convex is None and f.invex is None and f.quasi_invex is None and f.admissible is None


# --- sums ---------------------------------------------------------------------


def test_admissible_sum_makes_invex_penalties_sum_invex():
    s = Sum(L1Penalty(0.1), QuasinormInvexPenalty(0.2, 0.5))
    assert s.admissible.source == "rule:admissible_sum"
    assert s.invex.source == "rule:admissible_invex"
    assert s.convex is None  # quasinorm is not convex
    tree = rules.explain(s.invex)
    assert "rule:admissible_invex" in tree and "rule:admissible_sum" in tree and "Definition 4" in tree


def test_nested_sums_derive_recursively():
    s = Sum(Sum(LogPenalty(0.1), LogMinusRationalPenalty(0.1)), TikhonovPenalty(1.0))
    assert s.admissible is not None and s.invex is not None


def test_manual_certificate_overrides_derived_one():
    s = Sum(TikhonovPenalty(1.0), TikhonovPenalty(2.0))
    manual = _cert()
    s._certify("convex", manual)
    assert s.convex is manual


def test_status_is_the_weakest_premise():
    weak = _Quadratic(admissible=_cert("evidence"))
    strong = _Quadratic(admissible=_cert("verified"))
    assert Sum(weak, strong).admissible.status == "evidence"
    assert Sum(strong, strong).admissible.status == "verified"


# --- composition with a linear map ---------------------------------------------


def test_convex_affine_composition():
    c = Composed(TikhonovPenalty(1.0), MatrixOperator(torch.randn(3, 5, dtype=torch.float64)), b=torch.ones(3))
    assert c.convex.source == "rule:convex_affine"


def test_affine_invex_construction_needs_c1_and_full_row_rank():
    g = _Quadratic(invex=_cert(), differentiable=_cert())  # invex + C^1, not certified convex
    mask = torch.tensor([True, False, True, True])
    assert Composed(g, Subsample(mask)).invex.source == "rule:affine_invex_construction"
    assert Composed(g, MatrixOperator(torch.randn(3, 4))).invex is None  # rank unknown
    no_c1 = _Quadratic(invex=_cert())
    assert Composed(no_c1, Subsample(mask)).invex is None  # Lemma 2 needs C^1


def test_admissible_1d_difference_is_invex():
    c = Composed(LogPenalty(0.1), FiniteDifference1D(8))
    assert c.invex.source == "rule:admissible_difference_1d"
    assert Composed(LogPenalty(0.1), FiniteDifference1D(8), b=torch.zeros(7)).invex is None  # Thm 4 has no b


def test_admissible_2d_tv_is_only_quasi_invex():
    c = Composed(LogPenalty(0.1), FiniteDifference2D())
    assert c.invex is None
    assert c.quasi_invex.source == "rule:admissible_tv_2d"
    assert Composed(L1Penalty(0.1), FiniteDifference2D()).convex is not None  # TV-L1 is convex


# --- max and networks -----------------------------------------------------------


def test_max_of_invex_is_quasi_invex():
    m = Max(LogPenalty(0.1), QuasinormInvexPenalty(0.1, 0.5))
    assert m.invex is None
    assert m.quasi_invex.source == "rule:max_quasi_invex"


def test_ix_sequential_uses_registered_rule():
    net = ixSequential(ixLinear(4, 6, 4, torch.tanh), ixLinear(4, 6, 2, torch.tanh))
    assert net.invex.source == "rule:full_rank_composition"
    assert net.invex.derived_from == (net[0].invex, net[1].invex)


# --- registry -------------------------------------------------------------------


@pytest.fixture
def scratch_registry():
    saved = list(rules.base._REGISTRY)
    yield
    rules.base._REGISTRY[:] = saved


def test_user_rule_fires_and_is_named(scratch_registry):
    class Marked(_Quadratic):
        pass

    rules.register(rules.Rule(
        name="marked_is_convex", produces="convex", statement="test", reference=_REF,
        premises=lambda obj: [obj.admissible] if isinstance(obj, Marked) else None,
    ))
    f = Marked(admissible=_cert())
    assert f.convex.source == "rule:marked_is_convex"


@pytest.mark.parametrize("kwargs, error", [
    ({"reference": None}, ValueError),
    ({"reference": Reference(authors="", title="t", venue="v", year=2020, locator="")}, ValueError),
    ({"produces": "smooth"}, ValueError),
    ({"name": "convex_sum"}, ValueError),  # duplicate
])
def test_register_rejects_bad_rules(scratch_registry, kwargs, error):
    base = dict(name="r", produces="convex", statement="s", reference=_REF, premises=lambda obj: None)
    base.update(kwargs)
    with pytest.raises(error):
        rules.register(rules.Rule(**base))


def test_self_referential_user_rule_does_not_recurse(scratch_registry):
    rules.register(rules.Rule(
        name="loop", produces="admissible", statement="s", reference=_REF,
        premises=lambda obj: [obj.admissible],
    ))
    assert _Quadratic().admissible is None


# --- combinator values and gradients -------------------------------------------


def test_composed_grad_matches_autograd():
    torch.manual_seed(0)
    H = MatrixOperator(torch.randn(4, 6, dtype=torch.float64))
    c = Composed(LogPenalty(0.3), H, b=torch.randn(4, dtype=torch.float64))
    x = torch.randn(6, dtype=torch.float64, requires_grad=True)
    c.value(x).backward()
    assert torch.allclose(x.grad, c.grad(x.detach()))


def test_max_takes_the_active_branch():
    f, h = TikhonovPenalty(1.0), L1Penalty(1.0)
    m = Max(f, h)
    small, big = torch.tensor([0.1, -0.2]), torch.tensor([3.0, -4.0])
    assert torch.equal(m.value(small), h.value(small)) and torch.equal(m.grad(small), h.grad(small))
    assert torch.equal(m.value(big), f.value(big)) and torch.equal(m.grad(big), f.grad(big))


# --- numerical verifier ---------------------------------------------------------


@pytest.mark.parametrize("penalty", [
    L1Penalty(0.3), TikhonovPenalty(2.0), QuasinormInvexPenalty(0.1, 0.2), QuasinormInvexPenalty(0.1, 0.9),
    LogPenalty(0.2), LogPenalty(0.2, beta=7.0), LogMinusRationalPenalty(0.4, beta=1.0),
    LogMinusRationalPenalty(0.4), LogMinusRationalPenalty(0.4, beta=50.0),
])
def test_every_admissible_certificate_passes_definition_4_spot_check(penalty):
    assert penalty.admissible is not None
    assert NumericalVerifier().check(penalty, "admissible").status == "evidence"


def test_verifier_rejects_non_admissible_profiles():
    class Profile:
        def __init__(self, s):
            self.profile = s

    v = NumericalVerifier()
    assert v.check(Profile(lambda w: w ** 3), "admissible") is None  # s/w^2 increasing
    assert v.check(Profile(lambda w: w + 1.0), "admissible") is None  # s(0) != 0
    assert v.check(Profile(lambda w: -w), "admissible") is None  # decreasing
    assert v.check(_Quadratic(), "admissible") is None  # no profile
    assert v.check(L1Penalty(1.0), "invex") is None  # unsupported property


def test_evidence_propagates_into_derived_certificates():
    class UserPenalty(_Quadratic):
        def profile(self, w):
            return torch.log1p(w * w)  # Cauchy-like profile

    p = UserPenalty()
    p._certify("admissible", NumericalVerifier().check(p, "admissible"))
    s = Sum(p, L1Penalty(0.1))
    assert s.invex.status == "evidence"


# --- solvers warn off the objective they minimize --------------------------------


def _messages(record):
    return [str(w.message) for w in record if "certificate" in str(w.message)]


def test_fista_silent_when_sum_is_certified():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        FISTA(TikhonovPenalty(1.0), L1Penalty(0.1), step=0.5, max_iter=2).run(torch.ones(3))
    assert _messages(record) == []


def test_admissible_tv_admm_warns_and_reports_quasi_invex():
    smooth = _Quadratic(convex=_cert())
    solver = LinearizedADMM(smooth, LogPenalty(0.1), D=FiniteDifference2D(), max_iter=2)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        solver.run(torch.zeros(5, 5))
    (msg,) = _messages(record)
    assert "smooth+penalty(D@x)" in msg and "quasi-invex (rule:admissible_tv_2d)" in msg


def test_fasta_silent_for_convex_smooth_with_linear_operator():
    smooth = _Quadratic(convex=_cert())
    A = MatrixOperator(torch.randn(4, 6, dtype=torch.float64))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        FASTA(smooth, L1Penalty(0.1), A=A, tau=0.1, max_iter=2).run(torch.zeros(6, dtype=torch.float64))
    assert _messages(record) == []
