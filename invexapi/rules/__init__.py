"""Cited rules that derive certificates (convex, invex, admissible, ...) for composites.

Combinators (:class:`~invexapi.functions.Sum`, :class:`~invexapi.functions.Composed`,
:class:`~invexapi.functions.Max`, :class:`~invexapi.nn.ixSequential`) ask this registry
for any certificate not attached manually. Users may :func:`register` their own
:class:`Rule`, which must cite a published :class:`~invexapi.metadata.Reference`.
"""

from . import builtin  # noqa: F401  (registers the library's rules)
from .base import PROPERTIES, Rule, derive, explain, register, registered

__all__ = ["Rule", "register", "registered", "derive", "explain", "PROPERTIES"]
