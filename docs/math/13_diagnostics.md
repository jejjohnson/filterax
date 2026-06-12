# Diagnostics & Likelihood

A filter that runs is not a filter that works. The diagnostics in
this chapter answer "is the assimilation statistically healthy?" —
they *detect* misspecification (of $P$, $R$, the inflation, the
ensemble size), they don't fix it. filterax groups them by cost:
per-cycle health checks, calibration assessments (rolling windows,
need a truth in twin experiments), and a-posteriori covariance
diagnosis (long accumulations). All are pure functions under
`filterax.utils.*` taking pre-computed arrays.

## Innovation statistics

Everything starts from the innovation — the part of the observation
the forecast did not predict:

$$
d = y - \mathcal{H}(\bar{x}),
\qquad
S = C^{HH} + R,
$$

where $S$ is the innovation covariance: ensemble-sampled
observation-space spread plus observation error. For a
correctly-specified filter, $d \sim \mathcal{N}(0, S)$ — every
diagnostic below is a different projection of that single statement.
Componentwise, the **normalised innovation** $d_i / \sqrt{S_{ii}}$
should be $\mathcal{N}(0,1)$; the fully whitened residual
$S^{-1/2} d$ should be $\mathcal{N}(0, I)$.

## χ² consistency

Contracting $d$ against $S^{-1}$ gives the Mahalanobis statistic
(Mehra 1970):

$$
\chi^2 = d^\top S^{-1} d,
\qquad
E[\chi^2] = N_y,
\qquad
\operatorname{Var}[\chi^2] = 2 N_y .
$$

`chi2_normalized` reports $\chi^2 / N_y$ — target $\approx 1$.
Persistent values $> 1$ mean the filter is overconfident
(underdispersive: inflate, chapter 12); $< 1$ means overdispersive.
Single-window values are noisy (null standard deviation
$\sqrt{2/N_y}$); track the running mean. The signature takes
$S^{-1} d$ pre-computed so the diagnostic composes with whichever
solver produced the analysis. This same ratio is the data-implied
factor in adaptive inflation (chapter 12).

## Spread-skill

In twin/OSSE experiments where the truth $x^*$ is known, a
well-calibrated ensemble has spread matching the error of its mean:

$$
\mathrm{SSR} =
\sqrt{\frac{1}{N_x}\sum_i \sigma_i^2}
\Bigg/
\sqrt{\frac{1}{N_x}\sum_i \big(\bar{x}_i - x_i^*\big)^2},
$$

with $\sigma_i$ the per-variable ensemble standard deviation.
$\mathrm{SSR} \approx 1$ is calibrated; $< 1$ underdispersive
(increase inflation); $> 1$ overdispersive. Collapsing
`ensemble_spread` is the earliest warning of divergence — it shows up
before the RMSE does.

## Rank histograms

The rank histogram (Hamill 2001) tests calibration of the whole
*distribution*, not just its second moment. For each variable $i$ at
each time $t$, count how many of the $N_e$ members fall below the
truth; the rank lands in $\{0, \dots, N_e\}$ ($N_e + 1$ bins). If
truth and members are exchangeable, ranks are uniform. The shapes
diagnose:

- **U-shaped** — truth often outside the envelope: underdispersive.
- **Dome-shaped** — truth always in the middle: overdispersive.
- **Sloped** — systematic bias.

`rank_histogram_chi2` turns flatness into a number:

$$
\chi^2 = \sum_k \frac{(O_k - E_k)^2}{E_k},
\qquad
E_k = \frac{N_{\text{total}}}{N_{\text{bins}}},
$$

distributed as $\chi^2(N_{\text{bins}} - 1)$ under the uniform null.

## CRPS

The Continuous Ranked Probability Score is a strictly proper scoring
rule for the full forecast distribution against a scalar observation:

$$
\mathrm{CRPS} = E|X - y| - \tfrac{1}{2} E|X - X'|,
$$

with $X, X'$ independent draws from the forecast ensemble. filterax
computes the second term by the sorted-ensemble identity (Hersbach
2000),

$$
\tfrac{1}{2} E|X - X'|
  = \frac{1}{N_e^2} \sum_{j} x_{(j)} \big( 2j - 1 - N_e \big),
$$

with $x_{(j)}$ the order statistics — $O(N_e \log N_e)$ per
observation. Lower is better; the units are those of the observed
variable. `crps_ensemble_batch` averages over an observation vector.

## Desroziers diagnostics

A long innovation record lets you estimate the error covariances *a
posteriori* (Desroziers et al. 2005). With forecast departures
$d_f = y - \mathcal{H}(\bar{x}^f)$ and analysis departures
$d_a = y - \mathcal{H}(\bar{x}^a)$, consistency of the analysis
implies the cross-statistics

$$
E\big[ d_f d_f^\top \big] \approx H P H^\top + R,
\qquad
E\big[ d_a d_f^\top \big] \approx R,
\qquad
E\big[ d_a d_a^\top \big] \approx R - H A H^\top,
$$

where $A$ is the analysis-error covariance. The middle identity is
the workhorse: `desroziers_R_estimate` accumulates
$\hat{R} = E[d_a d_f^\top]$ over time; disagreement with the
prescribed $R$ means the observation-error covariance is
misspecified. Accumulate 100+ cycles for stable estimates.

## Degrees of freedom for signal

How much did the observations actually constrain the analysis? With
an explicit gain (Cardinali et al. 2004),

$$
\mathrm{DFS} = \operatorname{tr}(K H) \in [0, N_y]
$$

— close to $N_y$ when observations dominate, close to 0 when the
prior does. Transform filters never form $K$, so `dfs_from_ensemble`
estimates the analogous quantity from forecast/analysis
perturbations, $\mathrm{DFS} \approx \operatorname{tr}(C^{af} /
\sigma_f^2)$ per variable — how much the analysis anomalies remain
driven by the forecast anomalies.

## Gaussian log-likelihood

The innovation likelihood is both a diagnostic and the training
signal for differentiable DA (chapter 14):

$$
\log p(y \mid \text{forecast})
  = -\tfrac{1}{2} \Big[
      N_y \log(2\pi) + \log\lvert S \rvert + d^\top S^{-1} d
    \Big].
$$

filterax builds $S$ as a `gaussx.LowRankUpdate` — base $R$ plus the
rank-$(N_e - 1)$ ensemble term $U U^\top$ with
$U = (HX)'^\top / \sqrt{N_e - 1}$ — so $\log|S|$ dispatches through
the matrix-determinant lemma and $S^{-1} d$ through the Woodbury
identity, both at $O(N_e^2 N_y + N_e^3)$ without materialising the
dense $(N_y, N_y)$ matrix. `innovation_statistics` bundles $d$, $S$,
the whitened residual (a true Cholesky whitening), and the
log-likelihood in one call.

## Effective ensemble size and weight entropy

For weighted-ensemble variants (particle filters, hybrid schemes),
two degeneracy monitors:

$$
N_{\mathrm{eff}} = \frac{1}{\sum_j w_j^2},
\qquad
H(w) = -\sum_j w_j \log w_j .
$$

Equal weights give $N_{\mathrm{eff}} = N_e$ and $H = \log N_e$;
$N_{\mathrm{eff}} \ll N_e$ (or $H \to 0$) signals weight collapse and
the need to resample.

## Implementation in filterax

```python
import jax
import jax.numpy as jnp
import lineax as lx

import filterax as flx
from filterax.utils import (
    chi2_normalized, crps_ensemble, rank_histogram, rank_histogram_chi2,
    spread_skill_ratio,
)

k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
N_e, N_x, N_y = 40, 6, 3

# Twin experiment: truth and members exchangeable draws around a centre,
# observation drawn from N(H x*, R) — a perfectly calibrated forecast.
centre = jax.random.normal(k1, (N_x,))
x_true = centre + jax.random.normal(k2, (N_x,))
particles = centre + jax.random.normal(k3, (N_e, N_x))
R = lx.DiagonalLinearOperator(0.5 * jnp.ones(N_y))
y = x_true[:N_y] + jnp.sqrt(0.5) * jax.random.normal(k4, (N_y,))

stats = flx.innovation_statistics(particles, y, lambda x: x[:N_y], R)
d, S = stats["innovation"], stats["innovation_cov"]
S_inv_d = jnp.linalg.solve(S.as_matrix(), d)
print("chi2 / N_y:", float(chi2_normalized(d, S_inv_d, N_y)))
print("log p(y | forecast):", float(stats["log_likelihood"]))
print("spread-skill ratio:", float(spread_skill_ratio(particles, x_true)))

# Rank histogram over 200 calibrated (ensemble, truth) pairs.
def draw(key):
    kc, kt, ke = jax.random.split(key, 3)
    c = jax.random.normal(kc, (N_x,))
    truth = c + jax.random.normal(kt, (N_x,))
    ens = c + jax.random.normal(ke, (N_e, N_x))
    return ens, truth

keys = jax.random.split(jax.random.key(1), 200)
ens_t, truth_t = jax.vmap(draw)(keys)
counts = rank_histogram(ens_t, truth_t)
print("rank-histogram chi2:", float(rank_histogram_chi2(counts)),
      "~ chi2 on", counts.shape[0] - 1, "dof")
print("CRPS:", float(crps_ensemble(particles[:, 0], x_true[0])))
```

```
chi2 / N_y: 2.609184980392456
log p(y | forecast): -7.225412368774414
spread-skill ratio: 0.6982089281082153
rank-histogram chi2: 39.08833312988281 ~ chi2 on 40 dof
CRPS: 1.795293927192688
```

The aggregated diagnostic is clean — the rank-histogram $\chi^2$ of
39.1 sits right at its 40-dof null expectation, confirming
calibration. The *single-window* statistics scatter exactly as the
nulls predict ($\chi^2/N_y$ has standard deviation $\approx 0.8$
here): per-cycle numbers are for *tracking*, conclusions need
windows.

## Where next

- [Chapter 12 — Inflation](12_inflation.md): what to do when
  $\chi^2/N_y > 1$ — and the adaptive scheme that consumes this
  statistic automatically.
- [Chapter 14 — Differentiable assimilation](14_differentiable.md):
  the log-likelihood above as a training loss.
- [Chapter 3 — Observation model](03_observation_model.md): where $R$
  and $\mathcal{H}$ come from; Desroziers tells you when they're
  wrong.
- [API: Diagnostics](../api/diagnostics.md) — the full
  `filterax.utils.*` listing with the three-phase cost pipeline.

## References

- Mehra, R. K. (1970). *On the Identification of Variances and
  Adaptive Kalman Filtering.* IEEE Trans. Automatic Control, 15(2),
  175–184.
- Hamill, T. M. (2001). *Interpretation of Rank Histograms for
  Verifying Ensemble Forecasts.* Mon. Wea. Rev., 129(3), 550–560.
- Hersbach, H. (2000). *Decomposition of the Continuous Ranked
  Probability Score for Ensemble Prediction Systems.* Wea.
  Forecasting, 15(5), 559–570.
- Gneiting, T. & Raftery, A. E. (2007). *Strictly Proper Scoring
  Rules, Prediction, and Estimation.* JASA, 102(477), 359–378.
- Desroziers, G., Berre, L., Chapnik, B., & Poli, P. (2005).
  *Diagnosis of observation, background and analysis-error statistics
  in observation space.* Q. J. R. Meteorol. Soc., 131(613),
  3385–3396.
- Cardinali, C., Pezzulli, S., & Andersson, E. (2004).
  *Influence-Matrix Diagnostic of a Data Assimilation System.*
  Q. J. R. Meteorol. Soc., 130(603), 2767–2786.
- Liu, J. S. & Chen, R. (1998). *Sequential Monte Carlo Methods for
  Dynamic Systems.* JASA, 93(443), 1032–1044.
