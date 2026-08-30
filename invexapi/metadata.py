"""Rich prose docstrings elsewhere in this library tend to tangle three different
kinds of information together: which source file/kernel an implementation was
transcribed from, why one design was chosen over a rejected alternative, and what
invariant must be preserved across future edits. This module gives each of those
its own small dataclass, plus a ``@documented`` decorator that attaches them to a
class/function as introspectable attributes and registers it so a tool can retrieve 
every documented item as plain JSON via ``dump_all_json()`` without parsing docstrings 
or importing this library at all (if it just wants a static snapshot generated once 
and shipped elsewhere).

Deliberately plain stdlib dataclasses not something being validated as external input, 
and invexapi's only required dependency is PyTorch.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional, Sequence

__all__ = [
    "Reference",
    "Provenance",
    "DesignDecision",
    "Invariant",
    "documented",
    "dump_all",
    "dump_all_json",
]


@dataclass(frozen=True)
class Reference:
    """A citation backing a mathematical or design claim."""

    authors: str
    title: str
    venue: str
    year: int
    locator: str  # e.g. "Lemma 1, item 5 (Eq. 10)", "Section 3.1.4"


@dataclass(frozen=True)
class Provenance:
    """Which source file(s)/kernel(s) an implementation was transcribed from."""

    files: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class DesignDecision:
    """A choice made, an alternative rejected, and why.

    ``reference`` is set when the decision is backed by a specific paper result
    (e.g. "this prox formula because Theorem 3 proves it's the global optimizer"),
    left ``None`` for plain engineering judgement calls.
    """

    choice: str
    rejected: str
    rationale: str
    reference: Optional[Reference] = None


@dataclass(frozen=True)
class Invariant:
    """A constraint that must be preserved across future edits."""

    statement: str


_REGISTRY: list[type | Callable[..., Any]] = []


def documented(
    *,
    provenance: Sequence[Provenance] = (),
    decisions: Sequence[DesignDecision] = (),
    invariants: Sequence[Invariant] = (),
    example: Optional[Callable[[], Any]] = None,
):
    """Class/function decorator attaching structured documentation.

    ``provenance``/``decisions``/``invariants`` describe the decorated class or
    function itself, not any particular instance, so they're stored as plain
    attributes rather than requiring instantiation (unlike
    :class:`invexapi.penalties.Certificate`, which genuinely can vary per instance).

    ``example``, if given, is a zero-argument callable building a representative
    instance purely so :func:`dump_all` can read its ``Certificate``s — only
    meaningful for classes (e.g. concrete ``Penalty``s) whose certificates don't
    depend on constructor arguments.
    """

    def wrap(obj):
        obj.provenance = tuple(provenance)
        obj.design_decisions = tuple(decisions)
        obj.invariants = tuple(invariants)
        obj._metadata_example = example
        _REGISTRY.append(obj)
        return obj

    return wrap


def _certificates_for(obj) -> dict[str, Any]:
    example = getattr(obj, "_metadata_example", None)
    if example is None:
        return {}
    instance = example()
    certs = {}
    for name in ("convex", "invex", "quasi_convex", "quasi_invex"):
        cert = getattr(instance, name, None)
        if cert is not None:
            certs[name] = asdict(cert)
    return certs


def dump_all() -> list[dict[str, Any]]:
    """Every ``@documented`` class/function as plain, JSON-serializable dicts."""
    entries = []
    for obj in _REGISTRY:
        entries.append(
            {
                "name": obj.__qualname__,
                "module": obj.__module__,
                "provenance": [asdict(p) for p in obj.provenance],
                "design_decisions": [asdict(d) for d in obj.design_decisions],
                "invariants": [asdict(i) for i in obj.invariants],
                "certificates": _certificates_for(obj),
            }
        )
    return entries


def dump_all_json(**kwargs: Any) -> str:
    """:func:`dump_all`, serialized to a JSON string."""
    return json.dumps(dump_all(), **kwargs)
