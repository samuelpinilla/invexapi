"""Smooth data-fidelity terms measuring agreement with observations."""

from .amplitude_hinge import AmplitudeHingeSquared
from .robust import (
    AdaptiveRobustLoss,
    CauchyLoss,
    GemanMcClureLoss,
    SquaredLogMinusRationalLoss,
    WelschLoss,
)
from .slack_amplitude import SlackAmplitudeLeastSquares

__all__ = [
    "AmplitudeHingeSquared",
    "SlackAmplitudeLeastSquares",
    "CauchyLoss",
    "GemanMcClureLoss",
    "WelschLoss",
    "AdaptiveRobustLoss",
    "SquaredLogMinusRationalLoss",
]
