"""Smooth data-fidelity terms measuring agreement with observations."""

from .robust import (
    AdaptiveRobustLoss,
    CauchyLoss,
    GemanMcClureLoss,
    SquaredLogMinusRationalLoss,
    WelschLoss,
)
from .slack_amplitude import SlackAmplitudeLeastSquares

__all__ = [
    "SlackAmplitudeLeastSquares",
    "CauchyLoss",
    "GemanMcClureLoss",
    "WelschLoss",
    "AdaptiveRobustLoss",
    "SquaredLogMinusRationalLoss",
]
