"""``Certificate``: a claim that some object provably has a property (convex,
invex, ...). Lives at the package root, since it's
consumed by both ``penalties/`` (``Loss``'s convex/invex/quasi_convex/quasi_invex)
and ``nn/`` (``ixLinear``/``ixConv2d``/``ixSequential``'s ``invex``) - it belongs to
neither specifically, the same reasoning ``metadata.py`` already documents for
``Reference``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .metadata import Reference

__all__ = ["Certificate"]


@dataclass(frozen=True)
class Certificate:
    """``status`` distinguishes a claim taken on faith (``"assumed"``) from one
    mechanically checked (``"verified"`` - the seam for a future Lean/mathlib4-backed
    verifier). ``source`` names what produced the certificate. ``reference`` is
    optional: a certificate can be attached with no citation, but every certificate
    backed by a paper result should have one.
    """

    status: Literal["assumed", "verified"]
    source: str
    reference: Optional[Reference] = None
