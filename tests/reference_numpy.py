"""Independent NumPy implementations of the penalties' prox and gradient formulas,
written elementwise and in float32 to mirror the papers' reference math as
literally as possible. No GPU dependency - used only as a correctness oracle for
the vectorized PyTorch implementations.
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
    """The paper kernel's gradient of the prox: the first-order approximation
    1 - lamb*q*(q-1)*|x|^(q-2) at the INPUT. Kept as a faithful transcription for
    reference only - the library's backward is the exact implicit-function gradient
    at the output, tested with gradcheck instead.

    Returns (grad_wrt_lamb_elementwise, grad_wrt_x).
    """
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


def log_minus_rational_prox_np(x: np.ndarray, lamb: float, beta: float = 2.0) -> np.ndarray:
    """Prox of lamb*(log(1+|x|) - |x|/(beta+beta|x|)): largest Cardano root, clamped at 0."""
    x = x.astype(np.float32)
    t = np.abs(x)
    sign = np.where(x < 0, -1.0, 1.0).astype(np.float32)

    a = beta
    b = 2.0 * beta - beta * t
    c = beta * lamb + beta - 2.0 * beta * t
    d = (beta - 1.0) * lamb - beta * t

    p = -b / (3.0 * a)
    qq = p ** 3.0 + (b * c - 3.0 * a * d) / (6.0 * a ** 2.0)
    r = c / (3.0 * a)

    aux = np.maximum(qq ** 2 + (r - p ** 2) ** 3, 0.0)
    sol = np.maximum(_cbrt_np(qq + np.sqrt(aux)) + _cbrt_np(qq - np.sqrt(aux)) + p, 0.0)

    return (sign * sol).astype(np.float32)


def log_prox_np(x: np.ndarray, lamb: float, beta: float = 1.0) -> np.ndarray:
    """Prox of (lamb/beta)*log(1+beta|x|) (the paper code's scaling): larger quadratic root, clamped at 0."""
    x = x.astype(np.float32)
    aux = np.maximum((beta * np.abs(x) + 1.0) ** 2 - 4.0 * lamb * beta, 0.0)
    sol = np.sign(x) * np.maximum((beta * np.abs(x) - 1.0 + np.sqrt(aux)) / (2.0 * beta), 0.0)
    return sol.astype(np.float32)


def llq_prox_np(x: np.ndarray, lamb: float, beta: float, q: float, tol: float = 1e-6, steps: int = 50) -> np.ndarray:
    """Prox of (lamb/beta)*log(1+beta|x|^q): bisection on [(lamb q)^(1/(2-q)), |x|], 0 if no sign change."""
    x = x.astype(np.float64)
    sign = np.where(np.signbit(x), -1.0, 1.0)
    u = np.abs(x)

    def pol(z):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.signbit(lamb * q * z ** (q - 1.0) / (1.0 + beta * z ** q) + z - u), -1.0, 1.0)

    a = np.full_like(u, (lamb * q) ** (1.0 / (2.0 - q)))
    b = u.copy()
    c = np.zeros_like(u)
    bracketed = pol(a) != pol(b)
    for _ in range(steps):
        c = np.where(bracketed, (a + b) / 2.0, c)
        if np.all(~bracketed | ((b - a) / 2.0 <= tol)):
            break
        same = pol(a) == pol(c)
        a = np.where(bracketed & same, c, a)
        b = np.where(bracketed & ~same, c, b)
    return sign * np.where(bracketed, c, 0.0)
