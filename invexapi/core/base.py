from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Protocol

import torch

from ..certificate import Certificate
from ..metadata import Invariant, Reference, documented

__all__ = ["Reference", "Certificate", "Loss", "Penalty", "Verifier", "ManualVerifier", "CERTIFIABLE"]

CERTIFIABLE = ("convex", "invex", "quasi_convex", "quasi_invex", "admissible", "differentiable")


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
    classes it provably belongs to, via certificate properties - ``convex``,
    ``invex``, ``quasi_convex``, ``quasi_invex``, ``admissible`` (ICLR 2024 Def. 4)
    and ``differentiable`` (C^1) - each an ``Optional[Certificate]``.

    A certificate attached with :meth:`_certify` always wins; otherwise the
    property asks :mod:`invexapi.rules` to derive one (e.g. ``admissible`` =>
    ``invex``, or a ``Sum`` of admissible parts), and is ``None`` when no cited
    rule applies.
    """

    def __init__(self) -> None:
        for name in CERTIFIABLE:
            setattr(self, f"_{name}", None)

    def _certify(self, property_name: str, certificate: Certificate) -> None:
        """Record that this instance provably has ``property_name``.

        Called from subclass ``__init__`` bodies with a fact known at construction
        time (e.g. cited from a paper), or by whoever proved a property of a
        composite - not meant to be called by optimizer code.
        """
        if property_name not in CERTIFIABLE:
            raise ValueError(f"unknown property: {property_name}")
        setattr(self, f"_{property_name}", certificate)

    def _certificate(self, name: str) -> Optional[Certificate]:
        manual = getattr(self, f"_{name}", None)
        if manual is not None:
            return manual
        from ..rules import derive

        return derive(self, name)

    @property
    def convex(self) -> Optional[Certificate]:
        return self._certificate("convex")

    @property
    def invex(self) -> Optional[Certificate]:
        return self._certificate("invex")

    @property
    def quasi_convex(self) -> Optional[Certificate]:
        return self._certificate("quasi_convex")

    @property
    def quasi_invex(self) -> Optional[Certificate]:
        # Invexity unconditionally implies quasi-invexity, so a certified (or
        # derived) `invex` certificate doubles as the quasi-invex one.
        manual = getattr(self, "_quasi_invex", None)
        if manual is not None:
            return manual
        invex = self.invex
        if invex is not None:
            return invex
        from ..rules import derive

        return derive(self, "quasi_invex")

    @property
    def admissible(self) -> Optional[Certificate]:
        return self._certificate("admissible")

    @property
    def differentiable(self) -> Optional[Certificate]:
        return self._certificate("differentiable")

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
