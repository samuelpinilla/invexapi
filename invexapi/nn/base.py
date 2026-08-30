"""Shared certificate machinery for invex neural network layers."""

from __future__ import annotations

from typing import Iterable, Optional, Protocol

from ..certificate import Certificate
from ..metadata import DesignDecision, Invariant, Reference

__all__ = [
    "Certificate",
    "Reference",
    "InvexCertified",
    "certify_composition",
    "LAYER_REFERENCE",
    "UNVERIFIED_INVARIANT",
    "NO_RESTRICTION_DECISION",
]

LAYER_REFERENCE = Reference(
    authors="Pinilla, Sanabria, Bi, Egiazarian",
    title="What Makes Neural Networks Trainable? Invexity as a Structural Design Principle in AI",
    venue="Research Square (preprint)",
    year=2025,
    locator="Theorem 9, Corollary 2, Theorem 8",
)

_COMPOSITION_REFERENCE = Reference(
    authors=LAYER_REFERENCE.authors,
    title=LAYER_REFERENCE.title,
    venue=LAYER_REFERENCE.venue,
    year=LAYER_REFERENCE.year,
    locator="Theorem 8",
)

UNVERIFIED_INVARIANT = Invariant(
    "the invex certificate on ixLinear/ixConv2d is assumed, not runtime-verified"
)

NO_RESTRICTION_DECISION = DesignDecision(
    choice="accept any activation callable and any in/hidden/out shape without validation",
    rejected="restricting to a fixed allowlist of activations or enforcing full-row-rank shapes",
    rationale="runtime checks here would be either expensive or easily invalidated by training",
    reference=LAYER_REFERENCE,
)


class InvexCertified(Protocol):
    """Anything exposing an ``.invex`` certificate."""

    @property
    def invex(self) -> Optional[Certificate]: ...


def certify_composition(children: Iterable[InvexCertified]) -> Optional[Certificate]:
    """Certified iff every child is certified, else ``None``."""
    children = list(children)
    if children and all(child.invex is not None for child in children):
        return Certificate(status="assumed", source="composition", reference=_COMPOSITION_REFERENCE)
    return None
