# invexapi guide

The full tour of what the library contains. The [README](README.md) has the short
version. Citation labels such as **[ICLR'24]** link to [REFERENCES.md](REFERENCES.md).

Everything listed here can also be queried from code, which is always current:

```python
import invexapi
invexapi.catalog()                       # every function class
invexapi.catalog(invex=True)             # ... certified invex at default parameters
invexapi.rules.registered()              # every certification rule, with its citation
invexapi.metadata.dump_all_json(indent=2)  # design decisions, provenance, invariants
```

## Functions, rules, solvers and operators

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
| `LogPenalty(lamb, beta=1)` | `λ·log(1+β\|x\|)` | invex (β=1, [NeurIPS'22](REFERENCES.md#ref-neurips22) Eq. 7) |
| `LogMinusRationalPenalty(lamb, beta=2)` | `λ·(log(1+\|x\|) − \|x\|/(β+β\|x\|))` | invex (β=2, [NeurIPS'22](REFERENCES.md#ref-neurips22) Eq. 10) |
| `LLqPenalty(lamb, q, beta=1)` | `(λ/β)·log(1+β\|x\|^q)` | admissible ⇒ invex |
| `TikhonovPenalty(lamb)` | `λ/2·‖x‖²` | convex, invex, quasi-convex |
| `L1Penalty(lamb)` | `λ·‖x‖₁` | convex, invex, quasi-convex |
| `PSDTracePenalty(lamb)` | `λ·tr(Z) + 𝟙{Z ⪰ 0}` | convex, invex, quasi-convex |
| `SymmetricNuclearPenalty(lamb)` | `λ·‖Z‖_* + 𝟙{Z = Zᵀ}` | convex, invex, quasi-convex |
| `Composed(g, H, b)` | `g(Hx − b)` | from rules |
| `Max(f, h)` | `max(f, h)` | from rules |
| `LinearTilt(base, c)` | `base(x) − Re⟨c, x⟩` | convex when `base` is (rule `convex_tilt`); never invex alone |
| `CauchyLoss(delta)` | `log(1 + x²/δ²)` | admissible ⇒ invex; C¹ ([ICLR'24](REFERENCES.md#ref-iclr24) Thm 5) |
| `GemanMcClureLoss(delta)` | `2x²/(x² + 4δ²)` | admissible ⇒ invex; C¹ ([ICLR'24](REFERENCES.md#ref-iclr24) Thm 5) |
| `WelschLoss(delta)` | `1 − exp(−x²/(2δ²))` | admissible ⇒ invex; C¹ ([ICLR'24](REFERENCES.md#ref-iclr24) Thm 5) |
| `AdaptiveRobustLoss(alpha, c)` | Barron-style adaptive loss | admissible ⇒ invex for α ≤ 2; C¹ ([ICLR'24](REFERENCES.md#ref-iclr24) Thm 5) |
| `SquaredLogMinusRationalLoss()` | `log(1+x²) − x²/(2x²+2)` | admissible ⇒ invex; C¹ ([ICLR'24](REFERENCES.md#ref-iclr24) Thm 5) |
| `SlackAmplitudeLeastSquares(b, slack)` | `½‖\|z\| + s − b‖²` (smooth term) | nothing (nonconvex) |
| `AmplitudeHingeSquared(b)` | `½‖max(\|z\| − b, 0)‖²` (smooth term) | convex, C¹ (½·dist² to the disks `\|z\| ≤ b`) |

Both log proxes and the LLq prox return the true global minimiser, including when
the prox subproblem is nonconvex. The robust losses have value and gradient only
(no prox): use them as data-fidelity terms. Because they are C¹ and admissible,
`Composed(CauchyLoss(1.0), H, b=y)` is certified invex whenever `H` is declared full
row rank (robust regression with a global-optimum certificate).

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

- convex implies invex ([Hanson'81](REFERENCES.md#ref-hanson81)); convex + convex is convex
  ([Boyd'04](REFERENCES.md#ref-boyd04) §3.2.1), and admissible + admissible is admissible, hence
  invex ([ICLR'24](REFERENCES.md#ref-iclr24) Thm 3).
- g(Hx − b) inherits convexity from g ([Boyd'04](REFERENCES.md#ref-boyd04) §3.2.2); g(Hx − b) is invex when g is invex and C¹
  and H has full row rank ([NeurIPS'22](REFERENCES.md#ref-neurips22) Lemma 2).
- An admissible g of a 1-D difference is invex ([ICLR'24](REFERENCES.md#ref-iclr24) Thm 4); of the 2-D
  difference stack it is only quasi-invex ([ICLR'24](REFERENCES.md#ref-iclr24) App. B.1).
- The max of two invex functions is quasi-invex ([ICLR'24](REFERENCES.md#ref-iclr24) Lemma 1).
- Invex layers compose (`ixSequential`; [Trainable'25](REFERENCES.md#ref-trainable25) Thm 8).

Invex + invex derives nothing, because it is not invex in general. Add your own
theorem with `rules.register(Rule(...))`; a rule without a published `Reference`
is rejected.

`NumericalVerifier().check(f, "admissible")` checks [ICLR'24](REFERENCES.md#ref-iclr24) Def. 4 numerically for
any separable function that exposes `profile(w)`. A passing check gives a
certificate with status `"evidence"`. That status carries into every certificate
derived from it, so a numerical check is never presented as a proof.

**Solvers** (`invexapi.optim`) - generic, decoupled from the penalties above:

- `GradientDescent` - with optional Armijo backtracking line search
- `FISTA` - accelerated proximal gradient for `smooth(x) + penalty(x)`
- `NonlinearCG` - Polak-Ribière+ with automatic restart
- `LinearizedADMM` - for `smooth(x) + penalty(D@x)` (e.g. total variation)
- `FASTA` ([FASTA](REFERENCES.md#ref-fasta)) - adaptive/accelerated forward-backward splitting for
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
| `FractionalFourier` | linear | fast discrete FrFT ([Ozaktas'96](REFERENCES.md#ref-ozaktas96); also the `frft` function) |
| `LiftedFrFTAmbiguity` | linear | lifted FrFT-based radar ambiguity function `Z -> Tr(B_{α,k} Z)` |
| `LiftedFROG` | linear | lifted SHG FROG trace `Z -> Tr(B_{τ,k} Z)`, `|·|²` at `Z = xxᵀ` |
| `FROG` | nonlinear | SHG FROG trace `x -> |Σₙ x[n] x[n+τ] e^{−2πink/N}|²` |
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
