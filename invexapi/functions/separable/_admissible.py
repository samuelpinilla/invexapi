"""The ``admissible`` certificate (ICLR 2024 Def. 4) shared by the separable penalties."""

from __future__ import annotations

from ...certificate import Certificate
from ...metadata import Reference

ADMISSIBLE_DEFINITION = dict(
    authors="Pinilla, S. and Thiyagalingam, J.",
    title="Global Optimality for Non-linear Constrained Restoration Problems via Invexity",
    venue="International Conference on Learning Representations (ICLR)",
    year=2024,
)


def admissible_certificate(profile: str) -> Certificate:
    """Certificate that ``g(x) = sum_i s(|x_i|)`` with ``s = profile`` meets Def. 4.

    Def. 4 asks s(0) = 0, s' > 0 on (0, inf) and s(w)/w^2 non-increasing; each
    caller's ``profile`` satisfies these by direct inspection (the function is not
    named as admissible in the reference itself), and
    ``tests/test_admissible.py`` spot-checks all three conditions numerically.
    """
    return Certificate(
        status="assumed",
        source="manual",
        reference=Reference(**ADMISSIBLE_DEFINITION, locator=f"Definition 4, with s(w) = {profile}"),
    )
