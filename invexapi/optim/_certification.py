from __future__ import annotations

import warnings

from ..metadata import DesignDecision, documented


def _known_notes(obj, label: str, depth: int = 0) -> list:
    """The strongest certified class of ``obj``, else of each of its ``parts`` (recursively)."""
    for name, shown in (("convex", "convex"), ("invex", "invex"), ("quasi_invex", "quasi-invex")):
        cert = getattr(obj, name, None)
        if cert is not None:
            return [f"{label} is certified {shown} ({cert.source})"]
    if depth >= 4:
        return []
    notes = []
    for i, part in enumerate(getattr(obj, "parts", ()) or ()):
        notes += _known_notes(part, f"{label}.parts[{i}]", depth + 1)
    return notes


@documented(
    decisions=[
        DesignDecision(
            choice="duck-type obj via getattr(obj, ..., None), don't require isinstance(obj, Loss)",
            rejected="requiring obj to be a Loss instance before checking certificates",
            rationale=(
                "optimizers stay duck-typed throughout this library; an object with "
                "no invex/convex attributes at all is treated identically to a Loss "
                "with those attributes set to None - no certificate either way"
            ),
        )
    ],
)
def warn_if_unproven(obj, label: str) -> None:
    """Warn if ``obj`` carries no certificate backing a global-optimum claim.

    Only convex/invex silence the warning (every stationary point is then a global
    minimum). A quasi-invex certificate does not guarantee that, but it - and the
    strongest class certified for each part of a composite - is reported in the
    message, so the user sees what *is* known.
    """
    if getattr(obj, "invex", None) is None and getattr(obj, "convex", None) is None:
        notes = _known_notes(obj, label)
        known = f" Known: {'; '.join(notes)}." if notes else ""
        warnings.warn(
            f"{label} carries no convex/invex certificate - convergence to a "
            f"global optimum is not guaranteed.{known}",
            stacklevel=3,
        )
