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
  A minimal, PyTorch-only toolkit for <b>invex</b> optimization: functions with
  proven optimality certificates, and the solvers that use them.
</p>

---

## Why invexity?

Invex functions are a strict generalization of convexity: every stationary point
is still a global minimum, but the function itself need not be convex. That gives
non-convex penalties (sparsity-promoting, non-smooth, highly structured) the same
global-optimality guarantee convex optimization enjoys - without paying for it
with local minima.

`invexapi` packages this idea as code: functions that carry a machine-checkable
**certificate** of which mathematical class they belong to (convex, invex,
quasi-convex, quasi-invex, admissible), a **rule engine** that certifies
combinations only through published theorems, and generic solvers that warn you
when no optimality guarantee applies.

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
solver = FISTA(DataFidelity(y), QuasinormInvexPenalty(lamb=0.1, q=0.5), step=1.0)
x_hat, history = solver.run(y.clone())
# FISTA warns here: no theorem certifies this particular sum as invex, so a
# global optimum is not guaranteed. The warning lists what *is* certified.
```

Any object exposing the right methods (`value`, `grad`, `prox`) works as a
`smooth`/`penalty`/`objective`; the built-in functions are one plug-in choice.

## Certificates and rules

Every certificate says where it comes from. A combination of functions is
certified only when a published theorem covers it, and `explain()` shows the chain:

```python
from invexapi import L1Penalty, LogPenalty, Sum, rules

obj = Sum(L1Penalty(0.1), LogPenalty(0.2, beta=3.0))
print(rules.explain(obj.invex))
# rule:admissible_invex [assumed] - Pinilla, S. and Thiyagalingam, J. (2024), Theorem 3 (i)
#   rule:admissible_sum [assumed] - ... Theorem 3 (ii)
#     manual [assumed] - ... Definition 4, with s(w) = lamb * w
#     manual [assumed] - ... Definition 4, with s(w) = lamb * log(1 + beta*w), any beta > 0
```

Invex + invex derives nothing, because it is not invex in general. You can add
your own theorem with `rules.register(...)`, but only with a published reference.
Numerical checks are marked `"evidence"` and never presented as proofs.

## What's inside

| Module | Contents | List it from code |
|---|---|---|
| `invexapi.functions` | invex penalties (quasinorm, log, log-minus-rational, LLq), convex ones (L1, Tikhonov, PSD trace), robust losses (Cauchy, Geman-McClure, Welsch, adaptive), combinators (`Sum`, `Composed`, `Max`) | `invexapi.catalog(invex=True)` |
| `invexapi.rules` | cited rules that certify combinations | `invexapi.rules.registered()` |
| `invexapi.optim` | `FISTA`, `LinearizedADMM`, `FASTA`, `GradientDescent`, `NonlinearCG`, `LevenbergMarquardt` | |
| `invexapi.operators` | linear/nonlinear forward maps with adjoints, algebra (`A @ B`, `A.H`), `norm()`, `check_adjoint()`; finite differences, convolution, FrFT | |
| `invexapi.nn` | invex network layers (`ixLinear`, `ixConv2d`, `ixSequential`) | |
| `invexapi.metadata` | design decisions and provenance as data | `invexapi.metadata.dump_all_json()` |

See the **[GUIDE](GUIDE.md)** for every function's formula and certificate, the
full list of rules, solver details and the operator reference.

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

Proxes are checked against NumPy transcriptions of the papers' kernels where those
are exact, and against brute-force search where they are not; gradients are checked
with `torch.autograd.gradcheck`. Nothing in this repo depends on CUDA.

## Examples

See `examples/` for denoising and deblurring scripts covering FISTA and
Linearized ADMM with total variation, using both the quasinorm and log invex
penalties, plus `wavemax_ambiguity_fasta.py`: lifted radar-waveform recovery
from an FrFT-based ambiguity function with FASTA. The ADMM/TV examples need a test image, fetched separately:

```bash
python examples/data/download.py
```

## Citing

invexapi implements results from these papers (full list, links and BibTeX in
[REFERENCES.md](REFERENCES.md); GitHub's "Cite this repository" uses
[CITATION.cff](CITATION.cff)):

- S. Pinilla, T. Mu, N. Bourne, J. Thiyagalingam. *Improved imaging by invex
  regularizers with global optima guarantees*. NeurIPS 2022.
- S. Pinilla, J. Thiyagalingam. *Global optimality for non-linear constrained
  restoration problems via invexity*. ICLR 2024.
- S. Pinilla, A. M. Sanabria, J. Bi, K. Egiazarian. *What makes neural networks
  trainable? Invexity as a structural design principle in AI*. Preprint, 2025.

## License

BSD 3-Clause, see [LICENSE](https://github.com/samuelpinilla/invexapi/blob/main/LICENSE).
