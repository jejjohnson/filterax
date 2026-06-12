# Ensemble Smoothers

A filter is causal: the analysis at time $t$ conditions on observations
up to $t$ only, approximating the **filtering density**
$p(x_t \mid y_{1:t})$. For reanalysis, parameter estimation, or any
retrospective study, the right object is the **smoothing density**

$$
p(x_t \mid y_{1:T}), \qquad t < T,
$$

which also lets observations *after* $t$ inform the estimate. A
smoother never makes the estimate worse in expectation — it conditions
on strictly more data — and the gain is largest mid-trajectory, where
plenty of future observations exist. Ensemble smoothers obtain it
cheaply: run the filter forward once, store its history, then sweep
backward, recombining stored ensembles. No adjoint model, no extra
forward-model runs.

## EnKS — forward filter, backward gain recursion

Let $X^a_t$ and $X^f_{t+1}$ be the stored analysis and forecast
ensembles (the forecast at $t{+}1$ is the propagation of the analysis
at $t$), with anomaly matrices $A_t$ and $F_{t+1}$ (rows are members,
as everywhere). The Ensemble Kalman Smoother (Evensen & van Leeuwen
2000) initialises $X^s_{T-1} = X^a_{T-1}$ and recurses backward with
the **smoother gain**

$$
G_t = \underbrace{\tfrac{1}{N_e-1} A_t^{\top} F_{t+1}}_{C^{af}_{t,t+1}}
      \;\Big(\underbrace{\tfrac{1}{N_e-1} F_{t+1}^{\top} F_{t+1}}_{C^{ff}_{t+1}}\Big)^{+},
$$

applied per member:

$$
X^s_t[j] = X^a_t[j] + \Big(G_t \big(X^s_{t+1}[j] - X^f_{t+1}[j]\big)^\top\Big)^\top .
$$

Read it as the Kalman update with the "observation" being the future:
the cross-covariance $C^{af}$ says how analysis errors at $t$
correlate with forecast errors at $t{+}1$, and the discrepancy
$X^s_{t+1} - X^f_{t+1}$ — what smoothing changed about the future — is
pulled back through it. The forecast covariance has rank
$\le N_e - 1$, so the inverse is Moore-Penrose; filterax computes it
through the $N_e \times N_e$ Gram matrix $F F^\top$ (never the
$N_x \times N_x$ covariance), giving $O\!\big(T (N_e^2 N_x + N_e^3)\big)$
for the entire backward pass. The whole sweep is a reversed
`jax.lax.scan` — JIT-compatible and differentiable.

## Ensemble RTS

The classical Rauch-Tung-Striebel smoother is the same backward
recursion phrased with the model's transition matrix:
$G_t = P^a_t M_t^\top (P^f_{t+1})^{-1}$. In the ensemble version the
cross-covariance $C^{af}_{t,t+1}$ *is* the sample estimate of
$P^a_t M_t^\top$ — the propagated members carry the linearised dynamics
implicitly — so for zero model error `EnsembleRTS` coincides with
`EnKS` (Evensen 2003 §5). filterax ships both names sharing one
implementation, so code can say which interpretation it means.

## The square-root smoother

`EnKS` applies a raw per-member correction, the backward analogue of
the stochastic update. `EnsembleSqrtSmoother` (Whitaker & Compo 2002;
Tippett et al. 2003) is the backward analogue of the *deterministic*
filters of chapters 6-7: the smoothed mean is updated exactly as in
EnKS, while the anomalies are transformed through a symmetric square
root in ensemble space,

$$
\Lambda = I + K_e \big(D^{\top} D - F^{\top} F\big) K_e^{\top},
\qquad K_e = (F F^{\top})^{+} F, \qquad
A^s_t = \Lambda^{1/2} A_t,
$$

where $D$ holds the smoothed anomalies from $t+1$. Same complexity,
same mean as EnKS; the perturbations differ only by an ensemble-space
rotation, and the symmetric choice avoids accumulating cross-member
coupling over many backward steps — the same argument that selects the
symmetric square root in the ETKF. Pair it with a square-root forward
filter (ETKF, EnSRF, ESTKF).

## Fixed-lag smoothing

The full EnKS needs the entire $T$-window history in memory and lets
arbitrarily old states feel arbitrarily new data. `FixedLagSmoother`
restricts the backward pass at anchor $s$ to the window
$[s,\, \min(T{-}1,\, s + L)]$:

- $L = 0$ reproduces the filter; $L \ge T - 1$ reproduces the EnKS.
- Batch cost $O\!\big(T \cdot L \cdot (N_e^2 N_x + N_e^3)\big)$;
  in a streaming setting only $L{+}1$ ensembles are held, states older
  than the lag are finalised and dropped.

The benefit saturates quickly: the correlation
$C^{af}_{t,t+\ell}$ decays with lead $\ell$ at the decorrelation rate
of the dynamics (fast, for chaotic systems), so a lag of a few
decorrelation times captures nearly all of the smoothing gain at a
fraction of the memory. Beyond that, sampled long-range correlations
are mostly noise — a long lag can even *degrade* a small ensemble,
which is localization-in-time by the same logic as chapter 8.

## IES — Gauss-Newton in ensemble space

The smoothers above refine a sequential filter's history. The
**Iterative Ensemble Smoother** (Chen & Oliver 2013; Evensen et al.
2019) attacks a single window head-on as an inverse problem
$y = G(\theta) + \eta$, where $G$ maps parameters (or an initial
state) to the whole observation record. Each iteration re-targets the
*initial* ensemble plus a Kalman correction evaluated at the current
iterate:

$$
\theta_{i+1}^{(j)} = (1 - \alpha)\, \theta_i^{(j)} + \alpha \Big[\theta_0^{(j)}
    + K_i \big(y + \epsilon^{(j)} - G(\theta_i^{(j)})\big)\Big], \qquad
K_i = C^{\theta G}_i \big(C^{GG}_i + \Gamma\big)^{-1},
$$

with perturbations $\epsilon^{(j)} \sim \mathcal{N}(0, \Gamma)$ redrawn
per iteration. This is a Gauss-Newton iteration for the regularised
least-squares problem, with the ensemble statistics standing in for the
Jacobian (compare EKI in chapter 9 — the anchor to $\theta_0^{(j)}$,
rather than a drift from the current iterate, is the distinguishing
feature, and it is what keeps the prior in the objective). $\alpha = 1$
is the pure Chen-Oliver update; $\alpha < 1$ damps it,
Levenberg-Marquardt style, for strongly nonlinear $G$. One iteration
on a linear $G$ reduces IES to the (ensemble) smoother update —
that's `test_ies_single_step_matches_numpy_reference`. Its home turf
is history matching and reservoir-type problems where one backward
pass over a sequential filter is not enough.

## Implementation in filterax

All backward smoothers consume the stacked
`forecast_history` / `analysis_history` $(T, N_e, N_x)$ produced by a
Layer-2 filter's `assimilate()`:

```python
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

import filterax as flx


class Identity(flx.AbstractDynamics):
    def __call__(self, state, t0, t1):
        return state


class ObserveFirstTwo(flx.AbstractObsOperator):
    def __call__(self, state):
        return state[:2]


# Forward pass: 4 assimilation windows of an ETKF.
particles = jr.normal(jr.key(0), (30, 4))
R = lx.DiagonalLinearOperator(0.2 * jnp.ones(2))
obs_seq = [
    (jnp.array([0.5, -0.5]), 1.0),
    (jnp.array([0.6, -0.4]), 2.0),
    (jnp.array([0.4, -0.6]), 3.0),
    (jnp.array([0.5, -0.5]), 4.0),
]
filt = flx.ETKF(dynamics=Identity(), obs_op=ObserveFirstTwo())
history = filt.assimilate(particles, obs_seq, R)

# Backward pass: refine the history into the smoothing density.
smoothed = flx.smoothers.EnKS().smooth(
    history.forecast_history, history.analysis_history
)
smoothed.smoothed_history.shape          # (4, 30, 4)

# Final time untouched (no future data there) …
jnp.allclose(smoothed.smoothed_history[-1], history.analysis_history[-1])  # True
# … and earlier times tighten: mean spread at t=0 drops 0.674 → 0.581.
history.analysis_history[0].std(axis=0, ddof=1).mean()
smoothed.smoothed_history[0].std(axis=0, ddof=1).mean()

# Bounded-memory variant: only look 2 windows ahead.
flx.smoothers.FixedLagSmoother(lag=2).smooth(
    history.forecast_history, history.analysis_history
)

# IES: a single-window inverse problem, iterated.
G = jnp.array([[1.0, 0.5], [-0.3, 1.2], [0.7, 0.1]])
y = G @ jnp.array([1.0, -1.0])
Gamma = lx.DiagonalLinearOperator(0.1 * jnp.ones(3))
init = 2.0 * jr.normal(jr.key(1), (50, 2))
ies = flx.smoothers.IES(n_iterations=5, step_size=1.0)
out = ies.solve(init, y, Gamma, lambda theta: G @ theta)
out.particles.mean(axis=0)               # ≈ [0.98, -0.87] → toward [1, -1]
```

`SmoothingResult.particles` is the terminal ensemble
(`smoothed_history[-1]`), matching the `AssimilationResult.particles`
convention so a smoother output chains into a follow-up forecast. The
backward pass works with *any* `AbstractSequentialFilter` history and
is JIT- and `grad`-compatible (chapter 14).

## Choosing a smoother

| Smoother | Use when |
|---|---|
| `EnKS` | Default reanalysis: one backward pass over a stored filter history |
| `EnsembleRTS` | Same algorithm; use the name when the RTS interpretation matters |
| `EnsembleSqrtSmoother` | Paired with a deterministic forward filter; long backward chains |
| `FixedLagSmoother` | Streaming / bounded memory; lag of a few decorrelation times |
| `IES` | Strongly nonlinear single-window inverse problems (history matching) |

## Where next

- [Chapter 6 — ETKF](06_etkf.md): the forward square-root transform
  the sqrt smoother mirrors.
- [Chapter 9 — Ensemble Kalman processes](09_ensemble_kalman_processes.md):
  EKI/EKS — IES's iterative siblings for parameter estimation.
- [Chapter 14 — Differentiable assimilation](14_differentiable.md):
  backpropagating through filter + smoother chains.
- [Chapter 16 — The assimilation cycle](16_assimilation_cycle.md):
  where `forecast_history` / `analysis_history` come from.
- [API: Smoothers](../api/smoothers.md) ·
  [API: Filters](../api/filters.md)

## References

- Evensen, G., & van Leeuwen, P. J. (2000). *An ensemble Kalman
  smoother for nonlinear dynamics.* MWR 128(6).
- Evensen, G. (2003). *The Ensemble Kalman Filter: theoretical
  formulation and practical implementation.* Ocean Dynamics 53(4).
- Whitaker, J. S., & Compo, G. P. (2002). *An ensemble Kalman smoother
  for reanalysis.* Proc. Symp. on Observations, Data Assimilation and
  Probabilistic Prediction.
- Cosme, E., Verron, J., Brasseur, P., Blum, J., & Auroux, D. (2012).
  *Smoothing problems in a Bayesian framework and their linear Gaussian
  solutions.* MWR 140(2).
- Chen, Y., & Oliver, D. S. (2013). *Levenberg-Marquardt forms of the
  iterative ensemble smoother for efficient history matching and
  uncertainty quantification.* Computational Geosciences 17(4).
- Evensen, G., Raanes, P. N., Stordal, A. S., & Hove, J. (2019).
  *Efficient implementation of an iterative ensemble smoother for
  data assimilation and reservoir history matching.* Frontiers in
  Applied Mathematics and Statistics 5.
