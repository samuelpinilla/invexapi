from __future__ import annotations

import warnings

from ..metadata import DesignDecision, documented


@documented(
    decisions=[
        DesignDecision(
            choice="duck-type obj via getattr(obj, ..., None), don't require isinstance(obj, Loss)",
            rejected="requiring obj to be a Loss instance before checking certificates",
            rationale=(
                "optimizers stay duck-typed throughout this library; an object with "
                "no invex/convex attributes at all is treated identically to a Loss "
                "with those attributes set to None — no certificate either way"
            ),
        )
    ],
)
def warn_if_unproven(obj, label: str) -> None:
    """Warn if ``obj`` carries no certificate backing a global-optimum claim."""
    if getattr(obj, "invex", None) is None and getattr(obj, "convex", None) is None:
        warnings.warn(
            f"{label} carries no convex/invex certificate — convergence to a "
            "global optimum is not guaranteed.",
            stacklevel=3,
        )
