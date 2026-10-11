from __future__ import annotations

from typing import Optional

from .. import metadata

__all__ = ["catalog"]


def catalog(
    *,
    convex: Optional[bool] = None,
    invex: Optional[bool] = None,
    quasi_convex: Optional[bool] = None,
    quasi_invex: Optional[bool] = None,
) -> list[type]:
    """List function classes by certified class, e.g. ``catalog(invex=True)``.

    Each filter is ``True`` (must carry that certificate), ``False`` (must not) or
    ``None`` (don't care). Certificates are read off each class's ``@documented``
    example instance, so they reflect the *default* parameters - e.g.
    ``LogPenalty`` is listed as invex because its default ``beta=1`` is certified,
    even though other ``beta`` are not. ``False`` means "no certificate", never
    "provably not in the class".
    """
    wanted = {"convex": convex, "invex": invex, "quasi_convex": quasi_convex, "quasi_invex": quasi_invex}
    found = []
    for obj in metadata._REGISTRY:
        example = getattr(obj, "_metadata_example", None)
        if example is None or not obj.__module__.startswith(__package__):
            continue
        instance = example()
        if all(want is None or (getattr(instance, name) is not None) == want for name, want in wanted.items()):
            found.append(obj)
    return found
