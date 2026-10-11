"""The rule registry: cited theorems that derive certificates from other certificates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

from ..certificate import Certificate, weakest_status
from ..core.base import CERTIFIABLE as PROPERTIES
from ..metadata import DesignDecision, Invariant, Reference, documented

__all__ = ["Rule", "register", "registered", "derive", "explain", "PROPERTIES"]

Premises = Callable[[Any], Optional[Sequence[Certificate]]]


@dataclass(frozen=True)
class Rule:
    """A published theorem ``premises => obj has <produces>``.

    ``premises(obj)`` returns ``None`` when the rule does not apply to ``obj`` (wrong
    shape of object, a structural hypothesis fails, or a premise certificate is
    missing), else the certificates it consumes - all non-``None``. Structural
    hypotheses that are not certificates (an operator's declared
    ``full_row_rank``, ``type(H) is FiniteDifference1D``) are checked inside
    ``premises`` and documented in ``statement``.
    """

    name: str
    produces: str
    statement: str
    reference: Reference
    premises: Premises


_REGISTRY: List[Rule] = []


def _is_published(reference: Any) -> bool:
    return (
        isinstance(reference, Reference)
        and all(str(getattr(reference, f)).strip() for f in ("authors", "title", "venue"))
        and isinstance(reference.year, int)
    )


@documented(
    decisions=[
        DesignDecision(
            choice="register() rejects any rule without a published Reference",
            rejected="allowing uncited user rules",
            rationale=(
                "a derived certificate is only as good as the theorem behind it; the "
                "same no-unpublished-sources policy already enforced for Provenance"
            ),
        )
    ],
)
def register(rule: Rule) -> Rule:
    """Add ``rule`` to the registry (rules are tried in registration order)."""
    if not isinstance(rule, Rule):
        raise TypeError(f"expected a Rule, got {type(rule).__name__}")
    if rule.produces not in PROPERTIES:
        raise ValueError(f"rule {rule.name!r} produces unknown property {rule.produces!r}")
    if not _is_published(rule.reference):
        raise ValueError(f"rule {rule.name!r} must cite a published Reference (authors, title, venue, year)")
    if any(r.name == rule.name for r in _REGISTRY):
        raise ValueError(f"a rule named {rule.name!r} is already registered")
    _REGISTRY.append(rule)
    return rule


def registered() -> List[Rule]:
    """Every registered rule, in the order they are tried."""
    return list(_REGISTRY)


# (id(obj), property) pairs currently being derived - stops a user rule whose
# premises (indirectly) ask for the property it produces from recursing forever.
_ACTIVE: set = set()


@documented(
    invariants=[
        Invariant(
            "a derived certificate's status is the weakest status among the "
            "certificates it was derived from (evidence < assumed < verified)"
        ),
        Invariant("derive() never raises for a non-matching rule; no match => None"),
    ],
)
def derive(obj: Any, property_name: str) -> Optional[Certificate]:
    """Certificate for ``obj`` having ``property_name``, from the first matching rule."""
    key = (id(obj), property_name)
    if key in _ACTIVE:
        return None
    _ACTIVE.add(key)
    try:
        for rule in _REGISTRY:
            if rule.produces != property_name:
                continue
            premises = rule.premises(obj)
            if not premises or any(p is None for p in premises):
                continue
            return Certificate(
                status=weakest_status(premises),
                source=f"rule:{rule.name}",
                reference=rule.reference,
                derived_from=tuple(premises),
            )
        return None
    finally:
        _ACTIVE.discard(key)


def explain(certificate: Optional[Certificate], indent: str = "") -> str:
    """A readable tree of how ``certificate`` was obtained, premises indented below."""
    if certificate is None:
        return f"{indent}(no certificate)"
    ref = certificate.reference
    cited = f" - {ref.authors} ({ref.year}), {ref.locator}" if ref is not None else ""
    lines = [f"{indent}{certificate.source} [{certificate.status}]{cited}"]
    for premise in certificate.derived_from:
        lines.append(explain(premise, indent + "  "))
    return "\n".join(lines)
