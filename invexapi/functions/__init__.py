"""Every loss/penalty in invexapi, organized by what the function *is*.

Which class (convex, invex, ...) a function belongs to is NOT encoded in this
layout - it lives on each instance's certificates (``f.convex``, ``f.invex``, ...),
since those classes form a lattice and can depend on constructor parameters.
Use :func:`catalog` to list functions by certified class.
"""

from .catalog import catalog
from .combinators import Composed, LinearTilt, Max, Sum
from .fidelity import SlackAmplitudeLeastSquares
from .separable import (
    L1Penalty,
    LogMinusRationalPenalty,
    LogPenalty,
    QuasinormInvexPenalty,
    TikhonovPenalty,
)
from .sets import PSDTracePenalty

__all__ = [
    "catalog",
    "L1Penalty",
    "LogPenalty",
    "LogMinusRationalPenalty",
    "QuasinormInvexPenalty",
    "TikhonovPenalty",
    "PSDTracePenalty",
    "SlackAmplitudeLeastSquares",
    "Sum",
    "Composed",
    "Max",
    "LinearTilt",
]
