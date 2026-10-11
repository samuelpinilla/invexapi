"""Shared root-finding for separable penalties whose prox is a polynomial root.

For a separable ``g(x) = sum_i phi(|x_i|)`` with ``phi`` increasing, the prox of
``lamb*g`` at ``x`` is ``sign(x) * z`` with ``z >= 0`` minimizing
``h(z) = lamb*phi(z) + 0.5*(z - |x|)^2``. Stationary points of ``h`` are roots of a
polynomial; when ``phi`` is nonconvex, ``h`` can have a positive local minimizer
that is *not* global, so the candidate root must still be compared with ``z = 0``.

Every branch is written with the "double-where" pattern (unselected branches see
safe dummy inputs) so autograd never multiplies a zero by an infinite local
derivative and produces NaN gradients.
"""

from __future__ import annotations

import math
from typing import Callable

import torch


def cbrt(v: torch.Tensor) -> torch.Tensor:
    """Real cube root, valid for negative inputs too (C's ``cbrtf`` semantics)."""
    return torch.sign(v) * v.abs().pow(1.0 / 3.0)


def largest_real_quadratic_root(a: float, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Largest real root of ``a z^2 + b z + c`` (``a > 0``); ``-inf`` where none exists."""
    disc = b * b - 4.0 * a * c
    real = disc >= 0
    safe = torch.where(real, disc, torch.ones_like(disc))
    root = (-b + safe.sqrt()) / (2.0 * a)
    return torch.where(real, root, torch.full_like(root, -math.inf))


def largest_real_cubic_root(
    a: float, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    """Largest real root of ``a z^3 + b z^2 + c z + d`` (``a != 0``).

    Cardano's formula where the cubic has one real root, the trigonometric form
    where it has three - Cardano with the discriminant clamped at zero (as the
    reference kernels do) is not a root in the three-real-root case.
    """
    B, C, D = b / a, c / a, d / a
    P = C - B * B / 3.0
    Q = 2.0 * B ** 3 / 27.0 - B * C / 3.0 + D
    disc = (Q / 2.0) ** 2 + (P / 3.0) ** 3

    one_root = disc > 0
    disc_safe = torch.where(one_root, disc, torch.ones_like(disc))
    sq = disc_safe.sqrt()
    y_cardano = cbrt(-Q / 2.0 + sq) + cbrt(-Q / 2.0 - sq)

    tiny = torch.finfo(P.dtype).tiny
    P_safe = torch.where(one_root, -torch.ones_like(P), torch.clamp(P, max=-tiny))
    m = torch.sqrt(-P_safe / 3.0)
    arg = torch.clamp((3.0 * Q / (2.0 * P_safe)) / m, -1.0, 1.0)
    arg = torch.where(one_root, torch.zeros_like(arg), arg)
    y_trig = 2.0 * m * torch.cos(torch.acos(arg) / 3.0)

    return torch.where(one_root, y_cardano, y_trig) - B / 3.0


def global_prox_magnitude(
    candidate: torch.Tensor,
    t: torch.Tensor,
    lamb: float,
    phi: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Pick ``argmin_{z in {max(candidate,0), 0}} lamb*phi(z) + 0.5*(z-t)^2``.

    ``phi(0)`` is assumed to be 0. Ties go to the nonzero candidate.
    """
    real = torch.isfinite(candidate) & (candidate > 0)
    z = torch.where(real, candidate, torch.zeros_like(candidate))
    h_z = lamb * phi(z) + 0.5 * (z - t) ** 2
    h_0 = 0.5 * t * t
    return torch.where(real & (h_z <= h_0), z, torch.zeros_like(z))
