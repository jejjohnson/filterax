---
status: draft
version: 0.1.0
---

# filterX x Ensemble DA Diagnostics: Gap Analysis

**Subject:** Evaluation metrics and diagnostics specific to ensemble data
assimilation. Pure functions operating on ensemble statistics and observations.

**Date:** 2026-04-03

---

## 1  Scope

Diagnostics for validating ensemble data assimilation systems: spread-skill
analysis, innovation statistics, rank histograms, consistency tests, and
proper scoring rules. All functions are pure -- they take ensemble arrays,
observations, and operator outputs, and return scalar or array diagnostics.

**In scope:** Ensemble DA evaluation metrics, filter consistency checks,
reliability diagnostics, proper scoring rules for ensemble forecasts.

**Out of scope:** GP regression metrics (see `pyrox.gp` metrics),
deterministic forecast verification, observation quality control,
adaptive tuning algorithms (inflation/localization optimization).

**Key principle:** These diagnostics answer "is the filter working correctly?"
They do not fix filter problems -- they detect them.

---

## 2  Gap Catalog

### Gap 1: Ensemble Spread

**Type:** Dispersion

**Math:** Per-variable ensemble spread (standard deviation of ensemble members
around the ensemble mean):

$$\sigma_{\text{spread},i} = \sqrt{\frac{1}{N_e - 1} \sum_{j=1}^{N_e} (x_i^{(j)} - \bar{x}_i)^2}$$

where $\bar{x}_i = \frac{1}{N_e} \sum_{j=1}^{N_e} x_i^{(j)}$ is the ensemble
mean for variable $i$. The scalar summary is the RMS spread over all variables:

$$\text{RMS-spread} = \sqrt{\frac{1}{N_x} \sum_{i=1}^{N_x} \sigma_{\text{spread},i}^2}$$

Track over assimilation cycles. A well-calibrated filter has spread $\approx$ RMSE
(the spread-skill relationship).

```python
def ensemble_spread(
    ensemble: Float[Array, "Ne Nx"],
) -> Float[Array, " Nx"]:
    """Per-variable ensemble spread (std dev). Shape (Nx,)."""
    ...

def rms_spread(
    ensemble: Float[Array, "Ne Nx"],
) -> Float[Array, ""]:
    """Root-mean-square spread over all variables. Scalar."""
    ...
```

**When to use:** Every assimilation cycle. Compare against RMSE to check
calibration. Decreasing spread over time signals filter divergence.

**Ref:** Whitaker, J. S. & Loughe, A. F. (1998). *The Relationship between
Ensemble Spread and Ensemble Mean Skill.* Monthly Weather Review, 126(12).

---

### Gap 2: RMSE (vs Truth)

**Type:** Point accuracy

**Math:** Root mean squared error of the ensemble mean against the true state
(available in twin/OSSE experiments):

$$\text{RMSE} = \sqrt{\frac{1}{N_x} \sum_{i=1}^{N_x} (\bar{x}_{a,i} - x_{\text{true},i})^2}$$

where $\bar{x}_a$ is the analysis ensemble mean and $x_{\text{true}}$ is the
true state. Can be computed for forecast or analysis, per variable or globally.

```python
def rmse_vs_truth(
    ensemble_mean: Float[Array, " Nx"],
    x_true: Float[Array, " Nx"],
) -> Float[Array, ""]:
    """RMSE of ensemble mean against true state. Scalar."""
    ...

def rmse_per_variable(
    ensemble_mean: Float[Array, "T Nx"],
    x_true: Float[Array, "T Nx"],
) -> Float[Array, " Nx"]:
    """Time-averaged RMSE per variable. Shape (Nx,)."""
    ...
```

**When to use:** Twin experiments / OSSEs where the true state is known.
The primary accuracy metric. Track over time to detect filter divergence
(RMSE growing without bound).

**Ref:** Standard. See Evensen, G. (2009). *Data Assimilation: The Ensemble
Kalman Filter.* Springer. Ch. 6.

---

### Gap 3: Rank Histogram

**Type:** Reliability

**Math:** For each variable $i$ at each time step, compute the rank of the
true value $x_{\text{true},i}$ among the $N_e$ ensemble members
$\{x_i^{(1)}, \ldots, x_i^{(N_e)}\}$. The rank $r_i \in \{1, \ldots, N_e+1\}$
counts how many ensemble members are smaller than the truth.

Aggregate ranks across all variables and times into a histogram with $N_e + 1$
bins. For a reliable ensemble, the histogram should be **uniform**.

Diagnostic shapes:
- **U-shaped:** underdispersive (spread too small, need more inflation)
- **Dome-shaped:** overdispersive (spread too large)
- **Skewed:** systematic bias in the ensemble mean

Statistical test: $\chi^2 = \sum_{k=1}^{N_e+1} \frac{(O_k - E_k)^2}{E_k}$
where $E_k = N_{\text{total}} / (N_e + 1)$ is the expected count per bin.

```python
def rank_histogram(
    ensemble: Float[Array, "Ne Nx"],
    x_true: Float[Array, " Nx"],
) -> Int[Array, "Ne1"]:
    """Rank histogram counts. Shape (Ne+1,). Should be approximately uniform."""
    ...

def rank_histogram_chi2(
    counts: Int[Array, "Ne1"],
) -> Float[Array, ""]:
    """Chi-squared test statistic for rank histogram flatness. Scalar."""
    ...
```

**When to use:** Aggregate over many assimilation cycles (at least 100+) for
meaningful statistics. The primary diagnostic for ensemble reliability.

**Ref:** Hamill, T. M. (2001). *Interpretation of Rank Histograms for Verifying
Ensemble Forecasts.* Monthly Weather Review, 129(3), 550--560.

---

### Gap 4: Innovation Statistics

**Type:** Consistency

**Math:** The innovation vector is the difference between observations and the
prior (forecast) mapped to observation space:

$$\mathbf{d}_f = \mathbf{y} - H\bar{\mathbf{x}}_f$$

For a correctly specified filter, the innovations should satisfy:

$$E[\mathbf{d}_f] = \mathbf{0}, \qquad \text{Cov}(\mathbf{d}_f) = HPH^T + R$$

The normalized innovation is:

$$\hat{d}_i = \frac{d_{f,i}}{\sqrt{(HPH^T + R)_{ii}}}$$

which should be $\mathcal{N}(0, 1)$ for each observation component.

```python
def innovation(
    y_obs: Float[Array, " Ny"],
    Hx_f: Float[Array, " Ny"],
) -> Float[Array, " Ny"]:
    """Innovation vector d = y - H x_f. Shape (Ny,)."""
    ...

def normalized_innovation(
    y_obs: Float[Array, " Ny"],
    Hx_f: Float[Array, " Ny"],
    innovation_var: Float[Array, " Ny"],
) -> Float[Array, " Ny"]:
    """Normalized innovation d / sqrt(diag(HPH^T + R)). Should be ~N(0,1)."""
    ...

def innovation_mean_test(
    innovations: Float[Array, "T Ny"],
) -> Float[Array, " Ny"]:
    """Time-averaged innovation per obs component. Should be ~0."""
    ...
```

**When to use:** Monitor every cycle. Non-zero mean innovations indicate
bias; innovation variance departing from $HPH^T + R$ indicates misspecified
error covariances.

**Ref:** Kalnay, E. (2002). *Atmospheric Modeling, Data Assimilation and
Predictability.* Cambridge University Press. Ch. 5.

---

### Gap 5: Desroziers Consistency Diagnostics

**Type:** Consistency

**Math:** Uses products of innovation vectors to diagnose error covariance
specification. Define the forecast and analysis departures:

$$\mathbf{d}_f = \mathbf{y} - H\bar{\mathbf{x}}_f, \qquad \mathbf{d}_a = \mathbf{y} - H\bar{\mathbf{x}}_a$$

Then the following expectations hold for a consistent system:

$$E[\mathbf{d}_a \mathbf{d}_f^T] \approx R \qquad \text{(a posteriori estimate of observation error covariance)}$$

$$E[\mathbf{d}_f \mathbf{d}_f^T] \approx HPH^T + R \qquad \text{(innovation covariance)}$$

$$E[\mathbf{d}_a \mathbf{d}_a^T] \approx R - HAH^T \qquad \text{(analysis residual covariance)}$$

The first relation provides a diagnostic estimate of $R$ that can be compared
to the assumed $R$. If $E[\mathbf{d}_a \mathbf{d}_f^T] \neq R_{\text{assumed}}$,
the observation error covariance is misspecified.

```python
def desroziers_R_estimate(
    d_f: Float[Array, "T Ny"],
    d_a: Float[Array, "T Ny"],
) -> Float[Array, "Ny Ny"]:
    """A posteriori estimate of R from E[d_a d_f^T]. Shape (Ny, Ny)."""
    ...

def desroziers_innovation_cov(
    d_f: Float[Array, "T Ny"],
) -> Float[Array, "Ny Ny"]:
    """Innovation covariance E[d_f d_f^T] ~ HPH^T + R. Shape (Ny, Ny)."""
    ...

def desroziers_analysis_residual_cov(
    d_a: Float[Array, "T Ny"],
) -> Float[Array, "Ny Ny"]:
    """Analysis residual covariance E[d_a d_a^T] ~ R - HAH^T. Shape (Ny, Ny)."""
    ...
```

**When to use:** A posteriori diagnostic. Requires accumulation over many
cycles (typically 100+). Compare $E[\mathbf{d}_a \mathbf{d}_f^T]$ to the
prescribed $R$ to detect observation error misspecification. Compare
$E[\mathbf{d}_f \mathbf{d}_f^T]$ to $HPH^T + R$ to check overall consistency.

**Ref:** Desroziers, G., Berre, L., Chapnik, B., & Poli, P. (2005).
*Diagnosis of Observation, Background and Analysis-Error Statistics in
Observation Space.* Quarterly Journal of the Royal Meteorological Society,
131(613), 3385--3396.

---

### Gap 6: Effective Ensemble Size

**Type:** Degeneracy

**Math:** For weighted ensembles (particle filters, importance-weighted EnKF),
the effective ensemble size measures weight concentration:

$$N_{\text{eff}} = \frac{1}{\sum_{j=1}^{N_e} w_j^2}$$

where $w_j$ are normalized importance weights ($\sum_j w_j = 1$).

- $N_{\text{eff}} = N_e$ when all weights are equal (no degeneracy)
- $N_{\text{eff}} = 1$ when one particle has all the weight (complete degeneracy)
- Common resampling threshold: $N_{\text{eff}} < N_e / 2$

```python
def effective_ensemble_size(
    weights: Float[Array, " Ne"],
) -> Float[Array, ""]:
    """Effective ensemble size from normalized weights. Scalar in [1, Ne]."""
    ...

def weight_entropy(
    weights: Float[Array, " Ne"],
) -> Float[Array, ""]:
    """Shannon entropy of weights: -sum(w log w). Max = log(Ne) for uniform."""
    ...
```

**When to use:** Every cycle for particle filters or any weighted ensemble.
$N_{\text{eff}} \ll N_e$ triggers resampling. Also useful for diagnosing
observation impact in localized EnKF.

**Ref:** Liu, J. S. & Chen, R. (1998). *Sequential Monte Carlo Methods for
Dynamic Systems.* Journal of the American Statistical Association, 93(443),
1032--1044.

---

### Gap 7: Chi-Squared Consistency Test

**Type:** Consistency

**Math:** The normalized innovation squared statistic:

$$\chi^2 = \mathbf{d}_f^T S^{-1} \mathbf{d}_f$$

where $S = HPH^T + R$ is the innovation covariance. Under correct filter
specification, $\chi^2 \sim \chi^2(N_y)$ with:

$$E[\chi^2] = N_y, \qquad \text{Var}(\chi^2) = 2 N_y$$

The normalized statistic $\chi^2 / N_y$ should be $\approx 1$:
- $\chi^2 / N_y \gg 1$: innovations too large (underestimated errors)
- $\chi^2 / N_y \ll 1$: innovations too small (overestimated errors)

```python
def chi2_consistency(
    d_f: Float[Array, " Ny"],
    S_inv_d: Float[Array, " Ny"],
) -> Float[Array, ""]:
    """Chi-squared statistic d^T S^{-1} d. Should be ~Ny."""
    ...

def chi2_normalized(
    d_f: Float[Array, " Ny"],
    S_inv_d: Float[Array, " Ny"],
    n_obs: int,
) -> Float[Array, ""]:
    """Normalized chi-squared: d^T S^{-1} d / Ny. Should be ~1."""
    ...
```

**When to use:** Every cycle. The most compact single-number consistency check.
Time series of $\chi^2 / N_y$ immediately reveals filter health.

**Ref:** Mehra, R. K. (1970). *On the Identification of Variances and Adaptive
Kalman Filtering.* IEEE Transactions on Automatic Control, 15(2), 175--184.

---

### Gap 8: Spread-Skill Ratio

**Type:** Calibration

**Math:** The ratio of ensemble spread to RMSE:

$$\text{SSR} = \frac{\text{RMS-spread}}{\text{RMSE}} = \frac{\sqrt{\frac{1}{N_x} \sum_i \sigma_{\text{spread},i}^2}}{\sqrt{\frac{1}{N_x} \sum_i (\bar{x}_{a,i} - x_{\text{true},i})^2}}$$

Interpretation:
- $\text{SSR} \approx 1$: well-calibrated ensemble
- $\text{SSR} < 1$: underdispersive (spread too small, errors larger than expected -- increase inflation)
- $\text{SSR} > 1$: overdispersive (spread too large, errors smaller than expected -- decrease inflation)

Track as a time series and as a scatter plot (spread vs. error per variable).

```python
def spread_skill_ratio(
    ensemble: Float[Array, "Ne Nx"],
    x_true: Float[Array, " Nx"],
) -> Float[Array, ""]:
    """Spread-skill ratio. Should be ~1 for calibrated filter. Scalar."""
    ...

def spread_skill_binned(
    spreads: Float[Array, " T"],
    rmses: Float[Array, " T"],
    n_bins: int = 10,
) -> tuple[Float[Array, " B"], Float[Array, " B"]]:
    """Binned spread vs. RMSE for spread-skill scatter plot."""
    ...
```

**When to use:** Summary diagnostic. Compute over a window of cycles.
Directly informs inflation tuning.

**Ref:** Fortin, V., Abaza, M., Anctil, F., & Turcotte, R. (2014). *Why
Should Ensemble Spread Match the RMSE of the Ensemble Mean?* Journal of
Hydrometeorology, 15(4), 1708--1713.

---

### Gap 9: CRPS (Ensemble Form)

**Type:** Proper scoring rule

**Math:** The Continuous Ranked Probability Score for an ensemble forecast:

$$\text{CRPS}(F, y) = E|X - y| - \frac{1}{2} E|X - X'|$$

For an ensemble $\{x^{(1)}, \ldots, x^{(N_e)}\}$ with equal weights:

$$\text{CRPS} = \frac{1}{N_e} \sum_{j=1}^{N_e} |x^{(j)} - y| - \frac{1}{2 N_e^2} \sum_{j=1}^{N_e} \sum_{k=1}^{N_e} |x^{(j)} - x^{(k)}|$$

The second term can be computed efficiently using sorted members:

$$\frac{1}{2 N_e^2} \sum_{j=1}^{N_e} \sum_{k=1}^{N_e} |x^{(j)} - x^{(k)}| = \frac{1}{N_e^2} \sum_{j=1}^{N_e} x_{(j)} (2j - 1 - N_e)$$

where $x_{(j)}$ are the sorted ensemble members. Total cost: $O(N_e \log N_e)$ per observation.

Lower is better. CRPS is a strictly proper scoring rule and has the same units
as the observed variable.

```python
def crps_ensemble(
    ensemble: Float[Array, " Ne"],
    y_obs: float,
) -> Float[Array, ""]:
    """CRPS for a single observation against an ensemble. Scalar."""
    ...

def crps_ensemble_batch(
    ensemble: Float[Array, "Ne Ny"],
    y_obs: Float[Array, " Ny"],
) -> Float[Array, ""]:
    """Mean CRPS over all observations. Scalar."""
    ...
```

**When to use:** When evaluating ensemble forecasts against observations.
Preferred over RMSE because it rewards both accuracy and calibration.
Univariate -- apply per observation component and average.

**Ref:** Gneiting, T. & Raftery, A. E. (2007). *Strictly Proper Scoring Rules,
Prediction, and Estimation.* Journal of the American Statistical Association,
102(477), 359--378.

---

### Gap 10: Degrees of Freedom for Signal (DFS)

**Type:** Information content

**Math:** The DFS measures how much information observations add to the
analysis relative to the prior:

$$\text{DFS} = \text{tr}(I - KH) = N_y - \text{tr}(KH)$$

equivalently:

$$\text{DFS} = \text{tr}(KH) = \text{tr}\!\left(\frac{\partial \bar{\mathbf{x}}_a}{\partial \mathbf{y}}\right)$$

where $K = PH^T(HPH^T + R)^{-1}$ is the Kalman gain. Interpretation:

- $\text{DFS} \approx N_y$: observations dominate the analysis (strong constraint)
- $\text{DFS} \approx 0$: prior dominates the analysis (weak observations)
- $0 \leq \text{DFS} \leq N_y$ always

For ensemble methods, $\text{DFS}$ can be estimated from the ensemble
perturbation update without forming $K$ explicitly:

$$\text{DFS} \approx \text{tr}\!\left(\frac{1}{N_e - 1} \sum_{j=1}^{N_e} \frac{(\mathbf{x}_a^{(j)} - \bar{\mathbf{x}}_a)(\mathbf{x}_f^{(j)} - \bar{\mathbf{x}}_f)^T}{\sigma_f^2}\right)$$

```python
def dfs_from_gain(
    K: Float[Array, "Nx Ny"],
    H: Float[Array, "Ny Nx"],
) -> Float[Array, ""]:
    """DFS = tr(KH). Scalar in [0, Ny]."""
    ...

def dfs_from_ensemble(
    ensemble_f: Float[Array, "Ne Nx"],
    ensemble_a: Float[Array, "Ne Nx"],
) -> Float[Array, ""]:
    """Ensemble-based DFS estimate from forecast/analysis perturbations. Scalar."""
    ...
```

**When to use:** Observation impact studies. Identifies which observations
contribute most to the analysis. Useful for observation network design and
for diagnosing observation redundancy.

**Ref:** Cardinali, C., Pezzulli, S., & Andersson, E. (2004). *Influence-Matrix
Diagnostic of a Data Assimilation System.* Quarterly Journal of the Royal
Meteorological Society, 130(603), 2767--2786.

---

## 3  Diagnostic Workflow

A standard diagnostic pipeline for validating an ensemble DA system:

### Phase 1: Basic health checks (every cycle)

1. **Innovation statistics** (Gap 4): Compute $\mathbf{d}_f = \mathbf{y} - H\bar{\mathbf{x}}_f$. Check that normalized innovations are $\sim \mathcal{N}(0, 1)$.
2. **Chi-squared test** (Gap 7): Compute $\chi^2 / N_y$. Should be $\approx 1$. Flag cycles where it exceeds a threshold (e.g., > 3).
3. **Ensemble spread** (Gap 1): Track RMS-spread. If it collapses to zero, the filter has diverged.
4. **Effective ensemble size** (Gap 6): For particle filters, check $N_{\text{eff}}$ and trigger resampling if needed.

### Phase 2: Calibration assessment (accumulated over a window)

5. **Spread-skill ratio** (Gap 8): Compute over a rolling window (e.g., 50--100 cycles). Tune multiplicative inflation to drive SSR toward 1.
6. **RMSE vs truth** (Gap 2): Track analysis and forecast RMSE. The gap between them is the observation impact.
7. **Rank histogram** (Gap 3): Accumulate over 100+ cycles. Visual + chi-squared test for uniformity.

### Phase 3: Error covariance diagnosis (offline, long accumulation)

8. **Desroziers diagnostics** (Gap 5): Accumulate $\mathbf{d}_f$ and $\mathbf{d}_a$ over many cycles. Compare $E[\mathbf{d}_a \mathbf{d}_f^T]$ to prescribed $R$.
9. **DFS** (Gap 10): Assess observation information content. Identify redundant or ineffective observations.

### Phase 4: Forecast skill evaluation (verification against observations)

10. **CRPS** (Gap 9): Evaluate ensemble forecast quality as a proper scoring rule. Compare across different filter configurations.

### Summary table

| Diagnostic | Frequency | Requires truth? | Key threshold |
|---|---|---|---|
| Innovation mean/var | Every cycle | No | $E[d] \approx 0$ |
| $\chi^2 / N_y$ | Every cycle | No | $\approx 1$ |
| RMS spread | Every cycle | No | Non-collapsing |
| $N_{\text{eff}}$ | Every cycle | No | $> N_e / 2$ |
| Spread-skill ratio | Rolling window | Yes | $\approx 1$ |
| RMSE | Rolling window | Yes | Stable / decreasing |
| Rank histogram | 100+ cycles | Yes | Uniform ($\chi^2$ test) |
| Desroziers | 100+ cycles | No | $E[d_a d_f^T] \approx R$ |
| DFS | As needed | No | $0 \leq \text{DFS} \leq N_y$ |
| CRPS | Verification | Obs only | Lower is better |

---

## 4  References

1. Whitaker, J. S. & Loughe, A. F. (1998). *The Relationship between Ensemble Spread and Ensemble Mean Skill.* Monthly Weather Review, 126(12), 3292--3302.
2. Evensen, G. (2009). *Data Assimilation: The Ensemble Kalman Filter.* 2nd ed. Springer.
3. Hamill, T. M. (2001). *Interpretation of Rank Histograms for Verifying Ensemble Forecasts.* Monthly Weather Review, 129(3), 550--560.
4. Kalnay, E. (2002). *Atmospheric Modeling, Data Assimilation and Predictability.* Cambridge University Press.
5. Desroziers, G., Berre, L., Chapnik, B., & Poli, P. (2005). *Diagnosis of Observation, Background and Analysis-Error Statistics in Observation Space.* QJRMS, 131(613), 3385--3396.
6. Liu, J. S. & Chen, R. (1998). *Sequential Monte Carlo Methods for Dynamic Systems.* JASA, 93(443), 1032--1044.
7. Mehra, R. K. (1970). *On the Identification of Variances and Adaptive Kalman Filtering.* IEEE Trans. Automatic Control, 15(2), 175--184.
8. Fortin, V., Abaza, M., Anctil, F., & Turcotte, R. (2014). *Why Should Ensemble Spread Match the RMSE of the Ensemble Mean?* Journal of Hydrometeorology, 15(4), 1708--1713.
9. Gneiting, T. & Raftery, A. E. (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation.* JASA, 102(477), 359--378.
10. Cardinali, C., Pezzulli, S., & Andersson, E. (2004). *Influence-Matrix Diagnostic of a Data Assimilation System.* QJRMS, 130(603), 2767--2786.
