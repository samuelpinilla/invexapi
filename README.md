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

**Functions** (`invexapi.functions`) - loss/penalty terms plus a certificate of
what's provably known about each one. They are grouped by what the function *is*
(`separable/`, `sets/`, `fidelity/`, `combinators/`), not by its class. Being convex,
invex, etc. is recorded on each instance, because those classes overlap and can
depend on the parameters. To list functions by certified class:

```python
invexapi.catalog(invex=True)               # every function certified invex
invexapi.catalog(invex=True, convex=False) # invex but not certified convex
```

| Function | Form | Certified as |
|---|---|---|
| `QuasinormInvexPenalty(lamb, q)` | `λ·\|x\|^q` | invex |
| `LogPenalty(lamb, beta=1)` | `λ·log(1+β\|x\|)` | invex (β=1, NeurIPS'22 Eq. 7) |
| `LogMinusRationalPenalty(lamb, beta=2)` | `λ·(log(1+\|x\|) − \|x\|/(β+β\|x\|))` | invex (β=2, NeurIPS'22 Eq. 10) |
| `TikhonovPenalty(lamb)` | `λ/2·‖x‖²` | convex, invex, quasi-convex |
| `L1Penalty(lamb)` | `λ·‖x‖₁` | convex, invex, quasi-convex |
| `PSDTracePenalty(lamb)` | `λ·tr(Z) + 𝟙{Z ⪰ 0}` | convex, invex, quasi-convex |
| `Composed(g, H, b)` | `g(Hx − b)` | from rules |
| `Max(f, h)` | `max(f, h)` | from rules |
| `LinearTilt(base, c)` | `base(x) − Re⟨c, x⟩` | nothing (attach explicitly) |
| `SlackAmplitudeLeastSquares(b, slack)` | `½‖\|z\| + s − b‖²` (smooth term) | nothing (nonconvex) |

Both log proxes return the true global minimiser, including when the prox
subproblem is nonconvex.

**Rule engine** (`invexapi.rules`). A combination of functions is certified only
when a published theorem covers it. Each rule cites its source, and every derived
certificate records the certificates it was derived from:

```python
from invexapi import Composed, FiniteDifference1D, L1Penalty, LogPenalty, Sum, rules

obj = Sum(L1Penalty(0.1), LogPenalty(0.2, beta=3.0))
print(rules.explain(obj.invex))
# rule:admissible_invex [assumed] - Pinilla, S. and Thiyagalingam, J. (2024), Theorem 3 (i)
#   rule:admissible_sum [assumed] - ... Theorem 3 (ii)
#     manual [assumed] - ... Definition 4, with s(w) = lamb * w
#     manual [assumed] - ... Definition 4, with s(w) = lamb * log(1 + beta*w), any beta > 0

Composed(LogPenalty(0.1), FiniteDifference1D(64)).invex   # ICLR'24 Thm 4: invex
```

Built-in rules:

- convex + convex is convex, and admissible + admissible is admissible, hence
  invex (ICLR'24 Thm 3).
- g(Hx − b) inherits convexity from g; g(Hx − b) is invex when g is invex and C¹
  and H has full row rank (NeurIPS'22 Lemma 2).
- An admissible g of a 1-D difference is invex (ICLR'24 Thm 4); of the 2-D
  difference stack it is only quasi-invex (App. B.1).
- The max of two invex functions is quasi-invex (ICLR'24 Lemma 1).
- Invex layers compose (`ixSequential`).

Invex + invex derives nothing, because it is not invex in general. Add your own
theorem with `rules.register(Rule(...))`; a rule without a published `Reference`
is rejected.

`NumericalVerifier().check(f, "admissible")` checks ICLR'24 Def. 4 numerically for
any separable function that exposes `profile(w)`. A passing check gives a
certificate with status `"evidence"`. That status carries into every certificate
derived from it, so a numerical check is never presented as a proof.

**Solvers** (`invexapi.optim`) - generic, decoupled from the penalties above:

- `GradientDescent` - with optional Armijo backtracking line search
- `FISTA` - accelerated proximal gradient for `smooth(x) + penalty(x)`
- `NonlinearCG` - Polak-Ribière+ with automatic restart
- `LinearizedADMM` - for `smooth(x) + penalty(D@x)` (e.g. total variation)
- `FASTA` - adaptive/accelerated forward-backward splitting for
  `smooth(A@x) + penalty(x)` with BB step sizes and non-monotone backtracking
  (real or complex tensors)
- `LevenbergMarquardt` - nonlinear least squares `min_x 0.5*||F(x) - y||^2` for
  any differentiable `Operator`, with QR-solved damped Gauss-Newton steps

All of them warn (never error) when the objective they actually minimize
(e.g. `smooth + penalty(D@x)` for LinearizedADMM) has no convex/invex
certificate, since no global-optimum guarantee applies in that case. The warning
lists what *is* certified (e.g. a quasi-invex TV term).

**Operators** (`invexapi.operators`) - `Operator` is the base forward map
(`apply`); `LinearOperator` adds an `adjoint` (used by FASTA and LinearizedADMM)
and `NonlinearOperator` marks differentiable nonlinear maps (used by
LevenbergMarquardt through autograd):

| Operator | Kind | Map |
|---|---|---|
| `Identity` | linear | `x -> x` |
| `MatrixOperator(M)` | linear | `x -> M x` (dense) |
| `Diagonal(d)` | linear | `x -> d ⊙ x` |
| `Convolution(kernel, shape)` | linear | circular n-D convolution (FFT) |
| `Subsample(mask)` | linear | `x -> x[mask]` |
| `FiniteDifference1D(n)` | linear | `x -> x[1:] - x[:-1]` (full row rank) |
| `FiniteDifference2D` | linear | 2D total-variation gradient |
| `FractionalFourier` | linear | fast discrete FrFT (also the `frft` function) |
| `LiftedFrFTAmbiguity` | linear | lifted FrFT-based radar ambiguity function `Z -> Tr(B_{α,k} Z)` |
| `FrFTAmbiguity` | nonlinear | FrFT-based ambiguity function of a signal `x -> A[α,k]` |

Operators combine: `A @ B` (composition), `A + B`, `c * A`, `A.H` (adjoint), and
`A @ x` applies `A` to a tensor. The result is linear exactly when every part is.

```python
from invexapi.operators import Convolution, Subsample, check_adjoint

A = Subsample(mask) @ Convolution(kernel, (256, 256))  # blur, then keep observed pixels
check_adjoint(A)                    # inner-product identity, raises if wrong
step = 1.0 / A.norm() ** 2          # safe FISTA step for 0.5*||A x - y||^2
```

Each operator also records what is known about its structure: `full_row_rank`,
`injective`, `unitary` and `diagonal`. A value is `True`/`False` when it follows
exactly from the operator's construction, and `None` when unknown. Use
`.declare(full_row_rank=True)` to assert one yourself. Combining operators keeps
a property only when it provably carries over; otherwise the result is `None`.
(For a nonlinear operator these describe its Jacobian at every point.)

`in_shape`/`out_shape` give unbatched shapes (`None` = shape-agnostic). The FrFT
ambiguity operators take `pad` to zero-pad the signal before transforming, which
makes the fast FrFT much more accurate for off-centre or chirped signals.

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
penalties, plus `wavemax_ambiguity_fasta.py`: lifted radar-waveform recovery
from an FrFT-based ambiguity function with FASTA. The ADMM/TV examples need a test image, fetched separately:

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
