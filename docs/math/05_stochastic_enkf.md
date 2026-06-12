# The Stochastic EnKF

The stochastic ensemble Kalman filter (Evensen 1994; Burgers, van
Leeuwen & Evensen 1998) is the original and simplest member of the EnKF
family: compute one Kalman gain from the forecast ensemble, then update
every member against its own randomly perturbed copy of the
observation. This chapter states the update, explains why the
perturbations are not optional, and quantifies the price they carry.

## The perturbed-observation update

Given the forecast ensemble $X \in \mathbb{R}^{N_e \times N_x}$, the
observation $y$, and the gain $K = C^{xH} S^{-1}$ of chapter 4, each
member is updated as

$$
x_j^a = x_j^f + K\big(y_j - \mathcal{H}(x_j^f)\big),
\qquad
y_j = y + \epsilon_j,
\qquad
\epsilon_j \sim \mathcal{N}(0, R),
$$

for $j = 1, \dots, N_e$ — the perturbed observations of chapter 3.
Three points to note:

- **One gain, $N_e$ innovations.** $K$ is computed once from the
  forecast statistics; only the per-member innovation
  $y_j - \mathcal{H}(x_j^f)$ varies across members.
- **Per-member innovations, not the mean innovation.** Each member is
  pulled toward the data from *its own* position in observation space,
  so the update reshapes the whole ensemble, not just its mean.
- **Independent perturbations.** The $\epsilon_j$ are i.i.d. draws;
  reusing a draw across members (or across assimilation windows)
  silently breaks the covariance argument below.

Averaging over $j$ shows the analysis mean satisfies the standard
Kalman update $\bar{x}^a = \bar{x}^f + K\,d$ up to the
$O(1/\sqrt{N_e})$ sample mean of the perturbations, with
$d = y - \mathcal{H}(\bar{x})$ the usual innovation.

## Why the perturbations are necessary

The natural first attempt — update every member with the same,
unperturbed $y$ — is *wrong*, and the failure is systematic, not
random. Subtracting the mean update from the member update shows what
happens to the anomalies. Without perturbations (linear $H$ for
clarity):

$$
X'^a = X'^f (I - KH)^\top
\quad\Longrightarrow\quad
P^a_{\text{no pert}} = (I - KH)\, P\, (I - KH)^\top,
$$

whereas the correct Bayesian posterior covariance is

$$
P^a = (I - KH)\, P
= (I - KH)\, P\, (I - KH)^\top + K R K^\top.
$$

The unperturbed ensemble is missing exactly the $K R K^\top$ term — the
posterior uncertainty contributed by observation noise. Its spread is
therefore biased low at *every* analysis, the bias compounds over
cycles, and the filter spirals into the divergence failure mode of
chapter 1: too confident, deaf to data.

Burgers, van Leeuwen & Evensen (1998) showed that treating the
observation as a random variable restores the missing term. With
perturbed observations, the analysis anomalies become

$$
X'^a = X'^f (I - KH)^\top + E' K^\top,
$$

where $E'$ collects the centred perturbations $\epsilon_j$. Since the
perturbations are independent of the forecast errors and have
covariance $R$, taking expectations gives

$$
\mathbb{E}\big[P^a_{\text{pert}}\big]
= (I - KH)\, P\, (I - KH)^\top + K R K^\top
= (I - KH)\, P.
$$

The stochastic EnKF is thus a correct Monte Carlo sampler of the
Kalman posterior — in expectation, and exactly in the limit
$N_e \to \infty$.

## The price: sampling noise

"In expectation" is the catch. For finite $N_e$ the realised
perturbations have a sample mean and sample covariance that deviate
from $(0, R)$ by $O(1/\sqrt{N_e})$, and these deviations propagate
through $K$ into the analysis ensemble as extra Monte Carlo variance on
top of the sampling error the forecast ensemble already carries. With
small ensembles ($N_e \lesssim 50$, the operationally relevant regime)
this added noise is measurable: noisier analysis means, noisier
spread, more work for inflation.

The deterministic square-root filters (ETKF, EnSRF — chapters 6–7)
remove this noise source entirely: they achieve the correct posterior
covariance by an exact linear transform of the anomalies instead of by
random perturbation. The trade-off is not one-sided — the stochastic
filter's randomness makes it more robust to some nonlinear and
non-Gaussian effects, it never develops the preferred-direction
artefacts that deterministic transforms can, and it is the natural
bridge to randomised algorithms (EKI and friends). As a default,
though: prefer square-root filters at small $N_e$; the stochastic EnKF
remains the reference implementation and the simplest thing that is
actually correct.

## Algorithm

```
Inputs: X (N_e, N_x) forecast ensemble; y (N_y,) observation;
        H obs operator; R obs-noise operator; key PRNG key

Y   = H(X) row-wise                          # (N_e, N_y), chapter 3
K   = C^xH (C^HH + R)^{-1}                   # (N_x, N_y), chapter 4 (Woodbury)
Y_p = y + E,  E ~ N(0, R) per row            # (N_e, N_y) perturbed obs
X_a = X + (Y_p - Y) K^T                      # per-member update
return X_a, log p(y | forecast)              # likelihood from S = C^HH + R
```

Cost per analysis: $O(N_e \cdot \text{cost}(\mathcal{H}))$ for the
observation-space ensemble, $O(N_e^2 N_y + N_e^3)$ for the gain via
Woodbury, $O(N_e N_x N_y)$ to apply it.

## Implementation in filterax

```python
import jax
import jax.numpy as jnp
import lineax as lx
from filterax.filters import StochasticEnKF

N_e, N_x, N_y = 50, 6, 2
X = jax.random.normal(jax.random.key(0), (N_e, N_x)) + 2.0   # forecast

def obs_op(x):                       # H: observe the first two components
    return x[:2]

y = jnp.array([1.5, 2.5])
R = lx.DiagonalLinearOperator(0.25 * jnp.ones(N_y))

filt = StochasticEnKF(key=jax.random.key(0))
result = filt.analysis(X, y, obs_op, R, key=jax.random.key(42))

result.particles          # (50, 6) analysis ensemble
result.log_likelihood     # log p(y | forecast) under S = C^HH + R
```

`filterax.filters.StochasticEnKF` is the Layer-1 analysis step: one
forecast ensemble in, one posterior ensemble out, as an
`AnalysisResult`. The PRNG key stored on the filter is only the
*default*; pass a fresh `key=` per call (as above) or successive
windows will reuse identical observation perturbations — exactly the
correlated-draws mistake the theory forbids. The Layer-2 model
`filterax.StochasticEnKF` (top level, same name) wraps this step in the
full forecast → analyse → inflate loop and threads sub-keys per window
automatically; chapter 1's example shows the Layer-1 cycle written by
hand.

Everything inside `analysis` is assembled from the primitives of the
previous chapters: `kalman_gain` (chapter 4), `perturbed_observations`
(chapter 3), and the innovation likelihood under
$S = C^{HH} + R$. The step is `jit`-compatible and differentiable,
which is what makes the filter usable inside training loops.

## Where next

- [Chapter 4 — Kalman update](04_kalman_update.md): the gain this
  filter applies, and the Woodbury solve behind it.
- [Chapter 3 — Observation model](03_observation_model.md):
  perturbed observations and innovation statistics.
- [Chapter 1 — Problem setting](01_problem_setting.md): the
  forecast–analysis cycle this step slots into, and the divergence
  failure modes that inflation (chapter 12) and localization
  (chapter 11) address.
- [Filters API](../api/filters.md) — `filterax.filters.StochasticEnKF`
  (analysis step) and `filterax.StochasticEnKF` (cycling model).
- [Differentiable training API](../api/differentiable.md) — using the
  analysis log-likelihood as a learning signal.

## References

- Evensen, G. (1994). *Sequential data assimilation with a nonlinear
  quasi-geostrophic model using Monte Carlo methods to forecast error
  statistics.* JGR Oceans 99(C5).
- Burgers, G., van Leeuwen, P. J., & Evensen, G. (1998). *Analysis
  scheme in the ensemble Kalman filter.* MWR 126(6).
- Houtekamer, P. L., & Mitchell, H. L. (1998). *Data assimilation using
  an ensemble Kalman filter technique.* MWR 126(3).
- Tippett, M. K., Anderson, J. L., Bishop, C. H., Hamill, T. M., &
  Whitaker, J. S. (2003). *Ensemble square root filters.* MWR 131(7).
- van Leeuwen, P. J. (2020). *A consistent interpretation of the
  stochastic version of the Ensemble Kalman Filter.* QJRMS 146(731).
