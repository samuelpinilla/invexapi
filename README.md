<p align="center">
  <img src="https://raw.githubusercontent.com/samuelpinilla/invexapi/main/imgs/logo.png" alt="InvexAPI" width="480">
</p>

<p align="center">
  <a href="https://github.com/samuelpinilla/invexapi/blob/main/LICENSE"><img alt="License: BSD-3-Clause" src="https://img.shields.io/badge/license-BSD--3--Clause-blue.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="PyTorch only" src="https://img.shields.io/badge/backend-PyTorch-ee4c2c.svg">
</p>

<h1 align="center">invexapi</h1>

<p align="center">
  A minimal, PyTorch-only toolkit for <b>invex</b> optimization: penalties with
  proven optimality certificates, and the solvers that use them.
</p>

---

## Why invexity?

Invex functions are a strict generalization of convexity: every stationary point
is still a global minimum, but the function itself need not be convex. That gives
non-convex penalties (sparsity-promoting, non-smooth, highly structured) the same
global-optimality guarantee convex optimization enjoys - without paying for it
with local minima.

`invexapi` packages this idea as code: penalties that carry a machine-checkable
**certificate** of which mathematical class they belong to (convex, invex,
quasi-convex, quasi-invex), and generic solvers that read those certificates to
warn you when no optimality guarantee applies.

## Install

```bash
pip install -e .              # editable install
pip install -e ".[dev]"       # + pytest, numpy for the test suite
pip install -e ".[examples]"  # + deps used only by examples/
```

## Quick start

```python
import torch
from invexapi import QuasinormInvexPenalty
from invexapi.optim import FISTA

class DataFidelity:
    def __init__(self, y): self.y = y
    def value(self, x): return 0.5 * torch.sum((x - self.y) ** 2)
    def grad(self, x): return x - self.y

y = torch.randn(100)
smooth = DataFidelity(y)
penalty = QuasinormInvexPenalty(lamb=0.1, q=0.5)
solver = FISTA(smooth, penalty, step=1.0)

x_hat, history = solver.run(y.clone())
```

Any object exposing the right methods (`value`, `grad`, `prox`) works as a
`smooth`/`penalty`/`objective` - the built-in penalties are just one plug-in
choice.

## What's inside

**Penalties** (`invexapi.penalties`) - loss/penalty terms plus a certificate of
what's provably known about each one:

| Penalty | Form | Certified as |
|---|---|---|
| `QuasinormInvexPenalty(lamb, q)` | `λ·\|x\|^q` | invex |
| `LogInvexPenalty(lamb)` | `log(1+\|x\|) − \|x\|/(2+2\|x\|)` | invex |
| `TikhonovPenalty(lamb)` | `λ/2·‖x‖²` | convex, invex, quasi-convex |
| `L1Penalty(lamb)` | `λ·‖x‖₁` | convex, invex, quasi-convex |

Certificates are attached explicitly and never inferred - convexity composes
additively, but invexity does not, so a combined objective only carries a
certificate someone has actually proven for it.

**Solvers** (`invexapi.optim`) - generic, decoupled from the penalties above:

- `GradientDescent` - with optional Armijo backtracking line search
- `FISTA` - accelerated proximal gradient for `smooth(x) + penalty(x)`
- `NonlinearCG` - Polak-Ribière+ with automatic restart
- `LinearizedADMM` - for `smooth(x) + penalty(D@x)` (e.g. total variation)

All four warn (never error) when run on an objective without a convex/invex
certificate, since no global-optimum guarantee applies in that case.

**Linear operators** (`invexapi.penalties.operators`) - `Identity` and
`FiniteDifference2D` (2D total variation), each with a verified adjoint.

**Structured documentation** (`invexapi.metadata`) - design provenance,
rejected alternatives, and invariants as introspectable dataclasses rather than
prose, exportable as JSON for downstream tooling:

```python
import invexapi
print(invexapi.metadata.dump_all_json(indent=2))
```

## GPU support

Nothing in `invexapi` is device-specific - every operation derives its device
from its input tensors. Move your data to CUDA before calling a solver and
everything downstream follows:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_hat, history = solver.run(y.to(device))
```

## Testing

```bash
pytest
```

Penalty math is validated against independent NumPy reference implementations on
fixed-seed random inputs, with no CUDA dependency anywhere in this repo.

## Examples

See `examples/` for denoising and deblurring scripts covering FISTA and
Linearized ADMM with total variation, using both the quasinorm and log invex
penalties. The ADMM/TV examples need a test image, fetched separately:

```bash
python examples/data/download.py
```

## License

BSD 3-Clause, see [LICENSE](https://github.com/samuelpinilla/invexapi/blob/main/LICENSE).

## References

This library reproduces and generalizes results from the following papers:

1. Pinilla, S., Sanabria, A., Bi, J., & Egiazarian, K. (2025). *What makes neural
   networks trainable? Invexity as a structural design principle in AI*.
2. Pinilla, S., & Thiyagalingam, J. (2024). *Global optimality for non-linear
   constrained restoration problems via invexity*. International Conference on
   Learning Representations (ICLR), 2024, pp. 11990–12027.
3. Pinilla, S., Mu, T., Bourne, N., & Thiyagalingam, J. (2022). *Improved imaging
   by invex regularizers with global optima guarantees*. Advances in Neural
   Information Processing Systems (NeurIPS), 35, pp. 10780–10794.
4. Pinilla, S., Yeung, S.-L., & Thiyagalingam, J. (2024). *Global convergence of
   alternating direction method of multipliers for invex objective losses*. IEEE
   International Conference on Acoustics, Speech and Signal Processing (ICASSP)
   2024, pp. 9361–9365.
