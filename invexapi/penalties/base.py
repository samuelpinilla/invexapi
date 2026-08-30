from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Protocol

import torch

from ..certificate import Certificate
from ..metadata import DesignDecision, Invariant, Reference, documented

__all__ = ["Reference", "Certificate", "Loss", "Penalty", "Verifier", "ManualVerifier", "Sum"]


@documented(
    invariants=[
        Invariant(
            "convex/invex/quasi_convex/quasi_invex form a lattice, not a chain "
            "(Convex => Invex, Convex => Quasi-convex, both => Quasi-invex, but "
            "Invex and Quasi-convex are incomparable) - None means unproven, never "
            "false; never treat an unset certificate as evidence a property fails."
        ),
        Invariant(
            "optimizers in invexapi.optim never branch on a Loss's concrete class "
            "(only value/grad/prox via duck typing) but do read the certificate "
            "properties to decide whether a global-optimum warning is warranted."
        ),
    ],
)
class Loss(ABC):
    """Common interface for scalar loss/penalty terms ``g(x)``.

    Concrete subclasses supply ``value(x)`` (the scalar objective, for logging/
    stopping criteria) and ``grad(x)`` (the (sub)gradient, for gradient-based
    solvers). Beyond that, a ``Loss`` optionally declares which classical function
    classes it provably belongs to, via four properties - ``convex``, ``invex``,
    ``quasi_convex``, ``quasi_invex`` - each an ``Optional[Certificate]``.
    """

    def __init__(self) -> None:
        self._convex: Optional[Certificate] = None
        self._invex: Optional[Certificate] = None
        self._quasi_convex: Optional[Certificate] = None
        self._quasi_invex: Optional[Certificate] = None

    def _certify(self, property_name: str, certificate: Certificate) -> None:
        """Record that this instance provably has ``property_name``.

        Called from subclass ``__init__`` bodies with a fact known at construction
        time (e.g. cited from a paper) - not meant to be called by optimizer code.
        """
        if property_name not in ("convex", "invex", "quasi_convex", "quasi_invex"):
            raise ValueError(f"unknown property: {property_name}")
        setattr(self, f"_{property_name}", certificate)

    @property
    def convex(self) -> Optional[Certificate]:
        return self._convex

    @property
    def invex(self) -> Optional[Certificate]:
        return self._invex

    @property
    def quasi_convex(self) -> Optional[Certificate]:
        return self._quasi_convex

    @property
    def quasi_invex(self) -> Optional[Certificate]:
        # Invexity unconditionally implies quasi-invexity, so a certified `invex`
        # loss is quasi-invex too even without a separate _certify call - unlike
        # Sum's certificates, this fallback is mathematically unconditional, not an
        # approximation.
        return self._quasi_invex if self._quasi_invex is not None else self._invex

    @abstractmethod
    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Return the scalar value g(x)."""

    @abstractmethod
    def grad(self, x: torch.Tensor) -> torch.Tensor:
        """Return the (sub)gradient of g at x, same shape as x."""


class Penalty(Loss):
    """A :class:`Loss` that additionally supports a proximal operator.

    ``prox(x, step)`` is ``argmin_z step*g(z) + 0.5*||z-x||^2``, used by proximal
    solvers (:class:`~invexapi.optim.FISTA`); ``value``/``grad`` (inherited from
    ``Loss``) are used by gradient-based solvers (:class:`~invexapi.optim.
    GradientDescent`, :class:`~invexapi.optim.NonlinearCG`).
    """

    @abstractmethod
    def prox(self, x: torch.Tensor, step: float) -> torch.Tensor:
        """Return the proximal map of step*g evaluated at x, same shape as x."""


class Verifier(Protocol):
    """Something that can produce a :class:`Certificate` for a ``Loss`` property.

    This is the seam a future Lean/mathlib4-backed verifier plugs into: given a
    ``Loss`` (including a composite like :class:`Sum`) and a property name, decide
    whether that property provably holds and return the certificate, or ``None`` if
    it can't tell. Nothing in this codebase currently implements a verifier that
    computes anything new - see :class:`ManualVerifier`.
    """

    def check(self, obj: Loss, property_name: str) -> Optional[Certificate]: ...


class ManualVerifier:
    """A :class:`Verifier` that only reads back certificates already attached.

    A no-op passthrough today (it never *proves* anything, it just reports what a
    human already recorded via ``_certify``) - kept as a concrete implementation so
    code that consumes a ``Verifier`` has something to run against now, and so a
    real Lean-backed verifier can be swapped in later behind the same ``check``
    signature without touching call sites.
    """

    def check(self, obj: Loss, property_name: str) -> Optional[Certificate]:
        return getattr(obj, property_name)


@documented(
    decisions=[
        DesignDecision(
            choice="Sum's certificates default to None regardless of the parts' certificates",
            rejected="auto-deriving Sum's certificates from smooth's and penalty's certificates",
            rationale=(
                "convexity composes additively (convex + convex = convex) but "
                "invexity does not (invex + invex is not invex in general) - "
                "auto-deriving would be silently wrong for exactly the case this "
                "library cares about most, so no auto-derivation is done even for "
                "the convex case where it would happen to be sound."
            ),
        )
    ],
    invariants=[
        Invariant(
            "a certificate on the combined objective must be attached explicitly "
            "via sum_instance._certify(...) by whoever proved it (a human today, a "
            "Verifier later) - never inferred from smooth/penalty automatically."
        )
    ],
)
class Sum(Loss):
    """``smooth(x) + penalty(x)``, e.g. the combined objective FISTA minimizes."""

    def __init__(self, smooth: Loss, penalty: Penalty):
        super().__init__()
        self.smooth = smooth
        self.penalty = penalty

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.smooth.value(x) + self.penalty.value(x)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        return self.smooth.grad(x) + self.penalty.grad(x)
