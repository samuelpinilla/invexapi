"""Penalties acting entrywise, ``g(x) = sum_i phi(x_i)``, each with a closed-form prox."""

from .l1 import L1Penalty
from .llq import LLqPenalty
from .log import LogPenalty
from .log_minus_rational import LogMinusRationalPenalty
from .quasinorm import QuasinormInvexPenalty
from .tikhonov import TikhonovPenalty

__all__ = [
    "L1Penalty",
    "LogPenalty",
    "LLqPenalty",
    "LogMinusRationalPenalty",
    "QuasinormInvexPenalty",
    "TikhonovPenalty",
]
