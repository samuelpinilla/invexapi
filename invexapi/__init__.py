from importlib.metadata import version as _version

from . import metadata, nn, optim, penalties
from .certificate import Certificate
from .metadata import DesignDecision, Invariant, Provenance
from .optim import LinearizedADMM
from .penalties import (
    FiniteDifference2D,
    Identity,
    L1Penalty,
    LinearOperator,
    Loss,
    LogInvexPenalty,
    ManualVerifier,
    Penalty,
    QuasinormInvexPenalty,
    Reference,
    Sum,
    TikhonovPenalty,
    Verifier,
)

__version__ = _version("invexapi")

__all__ = [
    "optim",
    "penalties",
    "metadata",
    "nn",
    "Loss",
    "Penalty",
    "Certificate",
    "Reference",
    "Verifier",
    "ManualVerifier",
    "Sum",
    "QuasinormInvexPenalty",
    "LogInvexPenalty",
    "TikhonovPenalty",
    "L1Penalty",
    "LinearOperator",
    "Identity",
    "FiniteDifference2D",
    "LinearizedADMM",
    "Provenance",
    "DesignDecision",
    "Invariant",
]
