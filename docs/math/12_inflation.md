# Inflation

Every finite-ensemble Kalman filter systematically *underestimates*
its own uncertainty. Three effects compound:

1. **Finite $N_e$.** The analysis contracts the ensemble using a gain
   computed from the ensemble's own sampled covariance, and the
   sampling error preferentially over-contracts — the filter "trusts"
   noise modes it has just fit.
2. **Model error.** Members propagate through an imperfect model with
   no representation of the model's own error, so the forecast spread
   misses an entire error source.
3. **Localization side-effects.** Tapering (chapter 11) distorts
   analysis increments away from the optimal-gain geometry, which
   also removes variance.

Left untreated, the spread collapses over cycles, the gain
$K = C^{xH}(C^{HH}+R)^{-1} \to 0$, and the filter stops listening to
observations entirely — **filter divergence**: the forecast drifts
with the model while reporting near-zero uncertainty.

Inflation is the deliberate re-injection of spread. filterax ships
the classic trio (multiplicative, RTPS, RTPP — pure functions
delegating to the gaussx ensemble-DA primitives) plus three
filterax-specific advanced primitives (additive, adaptive,
Ledoit-Wolf).

## Multiplicative inflation

The simplest fix (Anderson & Anderson 1999): scale the anomalies
about the mean,

$$
x^{(j)} \leftarrow \bar{x} + \lambda \big( x^{(j)} - \bar{x} \big),
\qquad\text{i.e.}\qquad
X' \leftarrow \lambda X',
$$

so the covariance becomes $\lambda^2 P$ while the mean is untouched.
Typical operational values are $\lambda \in [1.01, 1.10]$. One knob,
applied uniformly — which is also its weakness: regions that are
densely observed (and genuinely contracted by data) get the same
boost as regions where the spread was already healthy.

## Relaxation methods: RTPS and RTPP

Relaxation methods are smarter: instead of inflating blindly, they
*relax the analysis back toward the forecast*, so the correction is
automatically largest exactly where the analysis contracted most.

**RTPS** — Relaxation to Prior **Spread** (Whitaker & Hamill 2012) —
works per variable on standard deviations. The target spread is a
convex blend of analysis and forecast spread,

$$
\sigma^{\text{target}}_i
  = (1 - \alpha)\, \sigma^a_i + \alpha\, \sigma^f_i,
$$

and each analysis anomaly is rescaled to hit it:

$$
x^{(j)}_{\text{relaxed}}
  = \bar{x}^a
  + \frac{\sigma^{\text{target}}_i}{\sigma^a_i}
    \big( x^{(j)}_a - \bar{x}^a \big).
$$

$\alpha = 0$ is the identity; $\alpha = 1$ restores the full forecast
spread. Because the factor is per-variable, RTPS is spatially
adaptive for free. It rescales each variable's spread but leaves the
analysis *correlation structure* intact.

**RTPP** — Relaxation to Prior **Perturbations** (Zhang et al. 2004)
— blends the anomaly matrices themselves:

$$
X'_{\text{relaxed}} = (1 - \alpha)\, X'_a + \alpha\, X'_f.
$$

Unlike RTPS, this mixes the *inter-variable correlation structure*
of forecast and analysis, not just the marginal spreads — useful when
the analysis update is suspected of damaging cross-correlations (e.g.
under aggressive localization), at the price of partially undoing the
correlation information the observations provided. Both are
mean-preserving and parameterised by a single $\alpha \in [0, 1]$.

## Additive inflation

When the dominant model-error mode is *not represented in the
ensemble at all* — structurally underdispersive in some subspace —
multiplying the existing anomalies cannot help: $\lambda \cdot 0 = 0$.
Additive inflation (Hamill & Whitaker 2005) injects noise with its
own covariance:

$$
X' \leftarrow X' + \varepsilon,
\qquad
\varepsilon^{(j)} \sim \mathcal{N}(0, Q_{\text{add}}),
$$

with $Q_{\text{add}}$ typically built from climatological
variability, lagged forecast differences, or stochastic-physics
tendency variances. `inflate_additive` centres the drawn noise so the
ensemble mean is exactly preserved, and a diagonal $Q_{\text{add}}$
never materialises. It is the one *stochastic* primitive in the set —
which matters for differentiable training (chapter 14).

## Adaptive inflation

Hand-tuning $\lambda$ is unsatisfying when the innovations themselves
tell you whether the filter is over- or under-dispersive. Anderson's
Bayesian scheme (Anderson 2009) treats $\lambda$ as a random
variable with Gaussian prior $\lambda \sim \mathcal{N}(\mu_\lambda,
\sigma^2_\lambda)$, updated each cycle by the normalised innovation.
The data-implied factor is the clamped Mahalanobis ratio

$$
\hat{\lambda}_{\text{obs}} = \operatorname{clip}\!\left(
    \frac{d^\top S^{-1} d}{N_y},\ \lambda_{\min},\ \lambda_{\max}
\right)
$$

— under correct specification $E[\chi^2 / N_y] = 1$ (chapter 13);
an underdispersive ensemble overshoots and pulls
$\hat{\lambda}_{\text{obs}}$ above 1. The posterior is the Gaussian
product of the prior with a likelihood centred at
$\hat{\lambda}_{\text{obs}}$ whose variance is the $\chi^2$ variance
$\sigma^2_{\text{obs}} = 2 / N_y$:

$$
\mu_{\text{post}} =
    \frac{\sigma^2_\lambda\, \hat{\lambda}_{\text{obs}}
          + \sigma^2_{\text{obs}}\, \mu_{\text{prior}}}
         {\sigma^2_\lambda + \sigma^2_{\text{obs}}},
\qquad
\sigma^2_{\text{post}} =
    \frac{\sigma^2_\lambda\, \sigma^2_{\text{obs}}}
         {\sigma^2_\lambda + \sigma^2_{\text{obs}}}.
$$

`inflate_adaptive` returns $(\mu_{\text{post}},
\sigma^2_{\text{post}})$ as JAX scalars, so the belief carries across
windows through `jax.jit` / `lax.scan`; applying the inflation is one
`inflate_multiplicative` call with $\mu_{\text{post}}$ as the factor.

## Ledoit-Wolf shrinkage

A different angle on the same disease: rather than inflating the
*ensemble*, regularise the *covariance estimate*. Ledoit-Wolf
shrinkage (Ledoit & Wolf 2004) replaces the sample covariance with a
convex combination toward a scalar-identity target,

$$
P_{\text{shrunk}}
  = (1 - \lambda^*)\, P_{\text{sample}} + \lambda^* \mu I,
\qquad
\mu = \frac{\operatorname{tr}(P_{\text{sample}})}{N_x},
$$

with the closed-form optimal intensity

$$
\lambda^* = \min\!\left( 1,\ \frac{b^2}{d^2} \right),
\qquad
d^2 = \lVert P_{\text{sample}} - \mu I \rVert_F^2,
\qquad
b^2 = \frac{1}{N_e^2} \sum_j
    \lVert x^{(j)} x^{(j)\top} - P_{\text{sample}} \rVert_F^2,
$$

minimising $E\lVert P_{\text{shrunk}} - P_{\text{true}} \rVert_F^2$.
The result is PSD and well-conditioned even when $N_e \ll N_x$, but
dense $(N_x, N_x)$ — call it only when $N_x$ is small enough to
materialise, e.g. parameter-space ensembles in EKI/EKS workflows
(chapter 9).

## Inflators in the L2 loop

The pure functions above are wrapped as `AbstractInflator` modules —
`MultiplicativeInflator`, `RTPS`, `RTPP`, `AdditiveInflator` — so an
inflation policy is a configuration object. In every L2 model
(`filterax.ETKF`, `LETKF`, …) the per-window loop is **forecast →
analysis → inflate**: the inflator receives both the analysis and the
forecast ensemble (so RTPS/RTPP can relax toward the prior), and its
output seeds the next forecast. `AdditiveInflator` additionally folds
the window index into its base PRNG key
(`jr.fold_in(base_key, step)`) so successive windows draw independent
perturbations; deterministic inflators ignore the kwarg.

## Implementation in filterax

```python
import jax
import jax.numpy as jnp

import filterax as flx

key = jax.random.key(0)
N_e, N_x = 20, 8
forecast = jax.random.normal(key, (N_e, N_x))
analysis = forecast * 0.4  # an over-tightened analysis

print("forecast spread:", float(flx.utils.rms_spread(forecast)))
print("analysis spread:", float(flx.utils.rms_spread(analysis)))

inflated = flx.inflate_multiplicative(analysis, 1.6)
relaxed_s = flx.RTPS(alpha=0.5)(analysis, forecast)
relaxed_p = flx.RTPP(alpha=0.5)(analysis, forecast)
print("multiplicative:", float(flx.utils.rms_spread(inflated)))
print("RTPS alpha=0.5:", float(flx.utils.rms_spread(relaxed_s)))
print("RTPP alpha=0.5:", float(flx.utils.rms_spread(relaxed_p)))

# Adaptive (Anderson 2009): carry (mu, sigma^2) across cycles.
d = jnp.array([1.5, -1.2, 0.9])              # an over-confident cycle
S = jnp.diag(jnp.array([0.5, 0.5, 0.5]))     # innovation covariance
mu, var = flx.inflate_adaptive(1.0, 0.05, d, S, max_factor=1.5)
print("posterior lambda:", float(mu), " var:", float(var))

# Ledoit-Wolf shrinkage of an anisotropic sample covariance.
scales = jnp.array([3.0, 2.0, 1.0, 0.5, 0.5, 0.3, 0.3, 0.2])
ens = jax.random.normal(key, (N_e, N_x)) * scales
P_shrunk, lam = flx.ledoit_wolf_shrinkage(ens)
print("shrinkage intensity:", round(lam, 3))
```

```
forecast spread: 0.9815067052841187
analysis spread: 0.39260268211364746
multiplicative: 0.6281642913818359
RTPS alpha=0.5: 0.6870546936988831
RTPP alpha=0.5: 0.6870546936988831
posterior lambda: 1.034883737564087  var: 0.04651162773370743
shrinkage intensity: 0.189
```

(RTPS and RTPP agree here because the analysis is an exact rescaling
of the forecast — they differ only when the analysis has *reshaped*
the anomalies, not just shrunk them.) The adaptive update nudges
$\lambda$ from 1.0 toward the data-implied value, but only as far as
the prior variance allows; the posterior variance shrinks for the
next cycle.

## Where next

- [Chapter 11 — Localization](11_localization.md): the companion fix
  for sampling noise — and one of the spread-collapse mechanisms
  inflation must offset.
- [Chapter 13 — Diagnostics & likelihood](13_diagnostics.md): the
  $\chi^2$ and spread-skill statistics that tell you *whether* to
  inflate, and that feed the adaptive update.
- [Chapter 14 — Differentiable assimilation](14_differentiable.md):
  why `AdditiveInflator` is rejected under `jax.grad` while the
  deterministic trio passes through.
- [Chapter 16 — The assimilation cycle](16_assimilation_cycle.md):
  where inflation sits in the L2 forecast–analyse–inflate loop.
- [API: Inflation](../api/inflation.md) — `inflate_multiplicative`,
  `inflate_rtps`, `inflate_rtpp`, `inflate_additive`,
  `inflate_adaptive`, `ledoit_wolf_shrinkage`, and the
  `AbstractInflator` wrappers.

## References

- Anderson, J. L. & Anderson, S. L. (1999). *A Monte Carlo
  implementation of the nonlinear filtering problem to produce
  ensemble assimilations and forecasts.* Mon. Wea. Rev., 127,
  2741–2758.
- Zhang, F., Snyder, C., & Sun, J. (2004). *Impacts of initial
  estimate and observation availability on convective-scale data
  assimilation.* Mon. Wea. Rev., 132, 1238–1253.
- Hamill, T. M. & Whitaker, J. S. (2005). *Accounting for the error
  due to unresolved scales in ensemble data assimilation: A
  comparison of different approaches.* Mon. Wea. Rev., 133,
  3132–3147.
- Whitaker, J. S. & Hamill, T. M. (2012). *Evaluating methods to
  account for system errors in ensemble data assimilation.* Mon.
  Wea. Rev., 140, 3078–3089.
- Anderson, J. L. (2009). *Spatially and temporally varying adaptive
  covariance inflation for ensemble filters.* Tellus A, 61, 72–83.
- Ledoit, O. & Wolf, M. (2004). *A well-conditioned estimator for
  large-dimensional covariance matrices.* J. Multivariate Anal., 88,
  365–411.
