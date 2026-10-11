"""``Certificate``: a claim that some object provably has a property (convex,
invex, ...). Lives at the package root, since it's
consumed by both ``core/`` (``Loss``'s convex/invex/quasi_convex/quasi_invex)
and ``nn/`` (``ixLinear``/``ixConv2d``/``ixSequential``'s ``invex``) - it belongs to
neither specifically, the same reasoning ``metadata.py`` already documents for
``Reference``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from .metadata import Reference

__all__ = ["Certificate", "STATUS_ORDER", "weakest_status"]


STATUS_ORDER = ("evidence", "assumed", "verified")


@dataclass(frozen=True)
class Certificate:
    """A claim that some object provably has a property.

    ``status`` says how the claim is backed: ``"assumed"`` (taken on faith, e.g.
    cited from a paper), ``"evidence"`` (numerically spot-checked by a
    :class:`~invexapi.core.NumericalVerifier` - never a proof), or ``"verified"``
    (mechanically checked - the seam for a future Lean/mathlib4-backed verifier).
    ``source`` names what produced it (``"manual"``, or ``"rule:<name>"`` for a
    certificate derived by :mod:`invexapi.rules`). ``reference`` is optional: a
    certificate can be attached with no citation, but every certificate backed by a
    paper result should have one. ``derived_from`` lists the certificates a rule
    consumed, so every derived claim can be audited back to its premises.
    """

    status: Literal["evidence", "assumed", "verified"]
    source: str
    reference: Optional[Reference] = None
    derived_from: Tuple["Certificate", ...] = ()


def weakest_status(certificates) -> str:
    """The least trustworthy status among ``certificates`` (evidence < assumed < verified)."""
    return min((c.status for c in certificates), key=STATUS_ORDER.index)
