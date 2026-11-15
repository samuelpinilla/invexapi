"""The library's own rules, each a published result.

Combinator types are imported inside the premise functions: ``invexapi.functions``
and ``invexapi.nn`` import ``invexapi.core``, which reaches this module lazily, so a
top-level import here would be circular.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..certificate import Certificate
from ..metadata import Reference
from .base import Rule, register

_BOYD: dict[str, Any] = dict(
    authors="Boyd, S. and Vandenberghe, L.",
    title="Convex Optimization",
    venue="Cambridge University Press",
    year=2004,
)
_HANSON = Reference(
    authors="Hanson, M. A.",
    title="On sufficiency of the Kuhn-Tucker conditions",
    venue="Journal of Mathematical Analysis and Applications",
    year=1981,
    locator="invexity definition: convex functions are invex with eta(x, y) = x - y",
)
_NEURIPS: dict[str, Any] = dict(
    authors="Pinilla, Mu, Bourne, Thiyagalingam",
    title="Improved Imaging by Invex Regularizers with Global Optima Guarantees",
    venue="NeurIPS",
    year=2022,
)
_ICLR: dict[str, Any] = dict(
    authors="Pinilla, S. and Thiyagalingam, J.",
    title="Global Optimality for Non-linear Constrained Restoration Problems via Invexity",
    venue="International Conference on Learning Representations (ICLR)",
    year=2024,
)
_LAYERS: dict[str, Any] = dict(
    authors="Pinilla, Sanabria, Bi, Egiazarian",
    title="What Makes Neural Networks Trainable? Invexity as a Structural Design Principle in AI",
    venue="Research Square (preprint)",
    year=2025,
)


def _all(*certificates: Optional[Certificate]) -> Optional[Sequence[Certificate]]:
    return list(certificates) if all(c is not None for c in certificates) else None


def _sum_parts(obj):
    from ..functions.combinators import Sum

    return (obj.smooth, obj.penalty) if isinstance(obj, Sum) else None


def _composed(obj):
    from ..functions.combinators import Composed

    return obj if isinstance(obj, Composed) else None


# --- implications on a single function (lattice) -----------------------------

register(Rule(
    name="convex_invex",
    produces="invex",
    statement="f convex => f invex (eta(x, y) = x - y)",
    reference=_HANSON,
    premises=lambda obj: _all(getattr(obj, "convex", None)),
))
register(Rule(
    name="admissible_invex",
    produces="invex",
    statement="f admissible (Def. 4) => f invex",
    reference=Reference(**_ICLR, locator="Theorem 3 (i)"),
    premises=lambda obj: _all(getattr(obj, "admissible", None)),
))
register(Rule(
    name="convex_quasi_convex",
    produces="quasi_convex",
    statement="f convex => f quasi-convex (convex sublevel sets)",
    reference=Reference(**_BOYD, locator="Section 3.4.1"),
    premises=lambda obj: _all(getattr(obj, "convex", None)),
))


# --- sums -------------------------------------------------------------------

def _sum_premises(prop):
    def premises(obj):
        parts = _sum_parts(obj)
        return None if parts is None else _all(*(getattr(p, prop, None) for p in parts))

    return premises


register(Rule(
    name="convex_sum",
    produces="convex",
    statement="f, g convex => f + g convex",
    reference=Reference(**_BOYD, locator="Section 3.2.1 (nonnegative weighted sums)"),
    premises=_sum_premises("convex"),
))
register(Rule(
    name="admissible_sum",
    produces="admissible",
    statement="f, g admissible (Def. 4) => f + g admissible (closure under nonnegative sums)",
    reference=Reference(**_ICLR, locator="Theorem 3 (ii)"),
    premises=_sum_premises("admissible"),
))


# --- linear tilts: base(x) - Re<c, x> --------------------------------------------

def _tilt_premises(obj):
    from ..functions.combinators import LinearTilt

    return _all(getattr(obj.base, "convex", None)) if isinstance(obj, LinearTilt) else None


register(Rule(
    name="convex_tilt",
    produces="convex",
    statement=(
        "f convex => f(x) - Re<c, x> convex (a linear function is convex, and the sum of "
        "convex functions is convex). Convexity only: a tilt keeps neither invexity nor "
        "quasi-convexity in general"
    ),
    reference=Reference(**_BOYD, locator="Sections 3.1.5 (linear functions) and 3.2.1 (nonnegative weighted sums)"),
    premises=_tilt_premises,
))


# --- composition with a linear map: g(Hx - b) -----------------------------------

def _convex_affine(obj):
    c = _composed(obj)
    return None if c is None else _all(getattr(c.g, "convex", None))


def _affine_invex(obj):
    c = _composed(obj)
    if c is None or getattr(c.H, "full_row_rank", None) is not True:
        return None
    return _all(getattr(c.g, "invex", None), getattr(c.g, "differentiable", None))


def _admissible_difference(kind):
    def premises(obj):
        from ..operators import FiniteDifference1D, FiniteDifference2D

        c = _composed(obj)
        want = {"1d": FiniteDifference1D, "2d": FiniteDifference2D}[kind]
        if c is None or type(c.H) is not want or c.b is not None:
            return None
        return _all(getattr(c.g, "admissible", None))

    return premises


register(Rule(
    name="convex_affine",
    produces="convex",
    statement="g convex => g(Hx - b) convex for any linear H",
    reference=Reference(**_BOYD, locator="Section 3.2.2 (composition with an affine mapping)"),
    premises=_convex_affine,
))
register(Rule(
    name="affine_invex_construction",
    produces="invex",
    statement=(
        "g invex and continuously differentiable, H full row rank => g(Hx - b) invex. "
        "Open item: the C^1 hypothesis excludes kinked penalties (L1, |x|^q, log(1+|x|)); "
        "a Clarke-subdifferential version would cover them but is not in the reference"
    ),
    reference=Reference(**_NEURIPS, locator="Lemma 2 (affine invex construction)"),
    premises=_affine_invex,
))
register(Rule(
    name="admissible_difference_1d",
    produces="invex",
    statement="g admissible, D the 1-D first-order difference (full row rank) => g(Dx) invex",
    reference=Reference(**_ICLR, locator="Theorem 4"),
    premises=_admissible_difference("1d"),
))
register(Rule(
    name="admissible_tv_2d",
    produces="quasi_invex",
    statement=(
        "g admissible, D the 2-D forward-difference stack => g(D_h x) + g(D_v x) quasi-invex "
        "(only quasi-invex: the stacked D is not full row rank). FiniteDifference2D's zeroed "
        "boundary rows add g(0) = 0 and its 1/sqrt(12) scale keeps g admissible"
    ),
    reference=Reference(**_ICLR, locator="Appendix B.1 (2D total-variation-like regularizer)"),
    premises=_admissible_difference("2d"),
))


# --- max ----------------------------------------------------------------------

def _max_premises(obj):
    from ..functions.combinators import Max

    if not isinstance(obj, Max):
        return None
    return _all(getattr(obj.f, "invex", None), getattr(obj.h, "invex", None))


register(Rule(
    name="max_quasi_invex",
    produces="quasi_invex",
    statement="f, h invex (not necessarily w.r.t. the same eta) => max(f, h) quasi-invex",
    reference=Reference(**_ICLR, locator="Lemma 1"),
    premises=_max_premises,
))


# --- invex network layers -------------------------------------------------------

def _sequential_premises(obj):
    invex_children = getattr(obj, "invex_children", None)
    if not callable(invex_children):
        return None
    children = list(invex_children())
    return _all(*(getattr(child, "invex", None) for child in children)) if children else None


register(Rule(
    name="full_rank_composition",
    produces="invex",
    statement=(
        "a composition of layers each invex with full-row-rank Jacobians is invex; "
        "the layers' certificates carry the rank assumption"
    ),
    reference=Reference(**_LAYERS, locator="Theorem 8"),
    premises=_sequential_premises,
))
