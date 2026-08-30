"""Direct NumPy transcriptions of the original CUDA kernels, kept close enough to
the source to eyeball-diff against it. No cupy/CUDA anywhere - used only to check
the PyTorch implementation against the paper's exact reference math.
"""

import numpy as np


def quasinorm_prox_np(x: np.ndarray, lamb: float, q: float) -> np.ndarray:
    x = x.astype(np.float32)
    absx = np.abs(x)
    sign = np.where(x < 0, -1.0, 1.0).astype(np.float32)

    tol = (lamb * q * (1.0 - q)) ** (1.0 / (2.0 - q))
    beta = (2.0 * lamb * (1.0 - q)) ** (1.0 / (2.0 - q))
    tau = beta + lamb * q * beta ** (q - 1.0)

    y = np.zeros_like(x)
    active = absx >= tol
    z = beta + (absx - beta) / 2.0
    with np.errstate(invalid="ignore"):
        for _ in range(4):
            z = absx - lamb * q * np.power(z, q - 1.0)

    y = np.where(active, sign * z, 0.0)
    y = np.where(active & (absx == tau), sign * beta, y)
    y = np.where(active & ((absx < tau) | (absx < beta)), 0.0, y)
    return y.astype(np.float32)


def quasinorm_grad_np(x: np.ndarray, grad_y: np.ndarray, lamb: float, q: float):
    """Returns (grad_wrt_lamb_elementwise, grad_wrt_x), matching gradG's (gx, gz)."""
    x = x.astype(np.float32)
    absx = np.abs(x)
    sign = np.where(x < 0, -1.0, 1.0).astype(np.float32)
    tol = (lamb * q * (1.0 - q)) ** (1.0 / (2.0 - q))
    active = absx >= tol

    grad_lamb = np.where(active, -q * np.power(absx, q - 1.0) * sign * grad_y, 0.0)
    grad_x = np.where(
        active,
        (1.0 - lamb * q * (q - 1.0) * np.power(absx, q - 2.0)) * grad_y,
        0.0,
    )
    return grad_lamb.astype(np.float32), grad_x.astype(np.float32)


def _cbrt_np(v: np.ndarray) -> np.ndarray:
    return np.sign(v) * np.abs(v) ** (1.0 / 3.0)


def log_prox_np(x: np.ndarray, lamb: float) -> np.ndarray:
    x = x.astype(np.float32)
    t = np.abs(x)
    sign = np.where(x < 0, -1.0, 1.0).astype(np.float32)

    a = 2.0
    b = 4.0 - 2.0 * t
    c = 2.0 * lamb + 2.0 - 4.0 * t
    d = 2.0 * lamb - lamb - 2.0 * t

    p = -b / (3.0 * a)
    qq = p ** 3.0 + (b * c - 3.0 * a * d) / (6.0 * a ** 2.0)
    r = c / (3.0 * a)

    aux = np.maximum(qq ** 2 + (r - p ** 2) ** 3, 0.0)
    beta = np.maximum(_cbrt_np(qq + np.sqrt(aux)) + _cbrt_np(qq - np.sqrt(aux)) + p, 0.0)

    return (sign * beta).astype(np.float32)
