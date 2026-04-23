---
status: draft
version: 0.1.0
---

# filterX — Sequential Ensemble Kalman Filters

**Subject:** Sequential ensemble Kalman filter algorithms for state estimation —
stochastic and deterministic square-root variants.

**Date:** 2026-04-03

---

## 1  Scope

Sequential ensemble filters for state estimation. All implement
`AbstractSequentialFilter` (see [components.md](../api/components.md)).

Each filter provides a single `analysis` step that assimilates observations into
a forecast ensemble. The forecast-analysis-inflate loop is handled by a higher
layer. Filters differ only in **how the ensemble is updated** given forecast
particles, observations, an observation operator, and observation noise.

**In scope:** Stochastic EnKF, ETKF, ETKF-Livings, EnSRF, EnSRF-Serial, ESTKF,
LETKF, parametric square-root Kalman filter.

**Out of scope:** Iterative processes (EKI, EKS — see processes.md), smoothers
(EnKS, RTS — see smoothers.md), particle filters.

---

## 2  Common Mathematical Framework

Every sequential ensemble Kalman filter follows the same four-step analysis
pattern. Given a forecast ensemble $\{x_f^{(j)}\}_{j=1}^{N_e}$, observations
$y \in \mathbb{R}^{N_y}$, observation operator $H$, and observation noise
covariance $R$:

**Step 1 — Forecast:** Propagate ensemble through dynamics (external).

$$x_f^{(j)} = \mathcal{M}(x_a^{(j)}), \quad j = 1, \ldots, N_e$$

**Step 2 — Ensemble statistics:**

$$\bar{x}_f = \frac{1}{N_e} \sum_j x_f^{(j)}, \qquad X' = \frac{1}{\sqrt{N_e - 1}} \bigl[x_f^{(1)} - \bar{x}_f \;\big|\; \cdots \;\big|\; x_f^{(N_e)} - \bar{x}_f\bigr]$$

$$Y' = \frac{1}{\sqrt{N_e - 1}} \bigl[H(x_f^{(1)}) - \overline{H(x_f)} \;\big|\; \cdots \;\big|\; H(x_f^{(N_e)}) - \overline{H(x_f)}\bigr]$$

**Step 3 — Kalman gain:**

$$C^{xH} = X' Y'^{\!\top}, \qquad C^{HH} = Y' Y'^{\!\top}$$

$$K = C^{xH} \bigl(C^{HH} + R\bigr)^{-1}$$

**Step 4 — Update ensemble:** This is where the filters diverge.

Two families:

| Family | Update strategy | Noise |
|--------|----------------|-------|
| **Stochastic** | Perturbed observations | Adds sampling noise to each member |
| **Deterministic** | Square-root transform | Transform anomaly matrix directly |

---

## 3  Gap Catalog

### Gap 1: StochasticEnKF — Stochastic Ensemble Kalman Filter

**Method family:** Stochastic (perturbed observations).

**Math:** Each ensemble member receives an independent observation perturbation:

$$\varepsilon^{(j)} \sim \mathcal{N}(0, R), \qquad j = 1, \ldots, N_e$$

$$x_a^{(j)} = x_f^{(j)} + K \bigl(y + \varepsilon^{(j)} - H x_f^{(j)}\bigr)$$

The perturbed observations ensure the analysis covariance is correct in
expectation:

$$P_a = (I - KH)\,P_f \quad \text{(in expectation over } \varepsilon \text{)}$$

Simple to implement but the perturbations introduce sampling noise that inflates
the analysis error, especially for small ensembles.

**Complexity:** $O(N_e^2 N_y + N_e^3)$ per analysis step. Dominated by the
ensemble cross-covariance computation and the $N_y \times N_y$ system solve.

```python
class StochasticEnKF(AbstractSequentialFilter):
    """Stochastic Ensemble Kalman Filter (Evensen 1994)."""
    seed: int = eqx.field(static=True, default=0)

    def analysis(
        self,
        particles: Float[Array, "N_e N_x"],
        obs: Float[Array, "N_y"],
        obs_op: AbstractObsOperator,
        obs_noise: AbstractLinearOperator,
        *, key: PRNGKey,
    ) -> AnalysisResult:
        # 1. Compute ensemble stats: x_bar, X_prime, Y_prime
        # 2. Kalman gain: K = C_xH @ solve(C_HH + R, ...)
        # 3. Sample perturbations: eps ~ N(0, R)
        # 4. Update: x_a[j] = x_f[j] + K @ (y + eps[j] - H x_f[j])
        ...
```

**Ref:** Evensen, G. (1994). *Sequential data assimilation with a nonlinear
quasi-geostrophic model using Monte Carlo methods to forecast error statistics.*
J. Geophys. Res., 99(C5), 10143--10162.

---

### Gap 2: ETKF — Ensemble Transform Kalman Filter

**Method family:** Deterministic (square-root transform).

**Math:** Work in the $N_e$-dimensional ensemble subspace. Define the transform
matrix via the observation-space precision:

$$\tilde{C} = (N_e - 1)\,I + Y'^{\!\top} R^{-1} Y' \in \mathbb{R}^{N_e \times N_e}$$

$$T = \tilde{C}^{-1}$$

The analysis weights and transform are:

$$\bar{w}_a = T \, Y'^{\!\top} R^{-1} (y - \bar{y}_f)$$

$$W_a = \sqrt{(N_e - 1)\,T}$$

The analysis ensemble is:

$$X_a = \bar{x}_f \mathbf{1}^{\!\top} + X'_f \bigl(\bar{w}_a \mathbf{1}^{\!\top} + W_a\bigr)$$

The symmetric square root $\sqrt{T}$ is computed via eigendecomposition of
$\tilde{C}$: if $\tilde{C} = U \Lambda U^{\!\top}$, then
$\sqrt{T} = U \Lambda^{-1/2} U^{\!\top}$.

**Complexity:** $O(N_e^2 N_y + N_e^3)$. The $N_e \times N_e$ eigendecomposition
is $O(N_e^3)$; forming $Y'^{\!\top} R^{-1} Y'$ is $O(N_e^2 N_y)$.

```python
class ETKF(AbstractSequentialFilter):
    """Ensemble Transform Kalman Filter (Bishop et al. 2001)."""

    def analysis(
        self,
        particles: Float[Array, "N_e N_x"],
        obs: Float[Array, "N_y"],
        obs_op: AbstractObsOperator,
        obs_noise: AbstractLinearOperator,
    ) -> AnalysisResult:
        # 1. Ensemble anomalies X', Y'
        # 2. C_tilde = (N_e - 1) I + Y'^T R^{-1} Y'
        # 3. Eigendecompose C_tilde = U Lambda U^T
        # 4. T = U Lambda^{-1} U^T
        # 5. w_bar = T @ Y'^T @ R^{-1} @ (y - y_bar)
        # 6. W_a = U @ diag(sqrt((N_e-1) / lambda)) @ U^T
        # 7. X_a = x_bar 1^T + X'_f @ (w_bar 1^T + W_a)
        ...
```

**Ref:** Bishop, C. H., Etherton, B. J., & Majumdar, S. J. (2001). *Adaptive
sampling with the ensemble transform Kalman filter. Part I: Theoretical aspects.*
Mon. Wea. Rev., 129, 420--436.

---

### Gap 3: ETKF_Livings — ETKF with Mean-Preserving Random Rotation

**Method family:** Deterministic (square-root transform with stochastic rotation).

**Math:** The symmetric square root in ETKF produces a unique but non-random
transform $W_a$. This can cause the ensemble to develop preferred directions
and lose rank over time.

Livings et al. replace $W_a$ with:

$$W_a^{\text{rot}} = W_a \, \Theta$$

where $\Theta \in O(N_e)$ is a random orthogonal matrix satisfying
$\Theta \mathbf{1} = \mathbf{1}$ (mean-preserving). This breaks the symmetry
while preserving the correct analysis covariance:

$$(W_a \Theta)(W_a \Theta)^{\!\top} = W_a W_a^{\!\top}$$

The random rotation is drawn once per analysis step. The constraint
$\Theta \mathbf{1} = \mathbf{1}$ is enforced by constructing $\Theta$ in the
$(N_e - 1)$-dimensional complement of $\mathbf{1}$.

**Complexity:** Same as ETKF: $O(N_e^2 N_y + N_e^3)$, plus $O(N_e^2)$ for the
random rotation.

```python
class ETKF_Livings(AbstractSequentialFilter):
    """ETKF with mean-preserving random rotation (Livings et al. 2008)."""

    def analysis(
        self,
        particles: Float[Array, "N_e N_x"],
        obs: Float[Array, "N_y"],
        obs_op: AbstractObsOperator,
        obs_noise: AbstractLinearOperator,
        *, key: PRNGKey,
    ) -> AnalysisResult:
        # 1-6. Same as ETKF
        # 7. Draw random orthogonal Theta with Theta @ 1 = 1
        # 8. W_a_rot = W_a @ Theta
        # 9. X_a = x_bar 1^T + X'_f @ (w_bar 1^T + W_a_rot)
        ...
```

**Ref:** Livings, D. M., Dance, S. L., & Nichols, N. K. (2008). *Unbiased
ensemble square root filters.* Physica D, 237(8), 1021--1028.

---

### Gap 4: EnSRF — Ensemble Square Root Filter

**Method family:** Deterministic (separate mean and perturbation updates).

**Math:** The mean and perturbations are updated independently:

**Mean update:**

$$\bar{x}_a = \bar{x}_f + K (y - H\bar{x}_f)$$

where $K = C^{xH}(C^{HH} + R)^{-1}$ is the standard Kalman gain.

**Perturbation update:**

$$X'_a = X'_f - \tilde{K}\,H\,X'_f$$

where $\tilde{K}$ is the "reduced" or "serial" Kalman gain. For the matrix
square root formulation:

$$\tilde{K} = P_f H^{\!\top} \bigl[(H P_f H^{\!\top} + R)^{1/2} \bigl((H P_f H^{\!\top} + R)^{1/2} + R^{1/2}\bigr)\bigr]^{-1}$$

This ensures $P_a = (I - KH)P_f$ exactly (no sampling noise).

**Complexity:** $O(N_e^2 N_y + N_y^3)$. The square root of $C^{HH} + R$ is
$O(N_y^3)$ unless done serially.

```python
class EnSRF(AbstractSequentialFilter):
    """Ensemble Square Root Filter (Whitaker & Hamill 2002)."""

    def analysis(
        self,
        particles: Float[Array, "N_e N_x"],
        obs: Float[Array, "N_y"],
        obs_op: AbstractObsOperator,
        obs_noise: AbstractLinearOperator,
    ) -> AnalysisResult:
        # 1. Ensemble stats: x_bar, X', Y'
        # 2. Standard Kalman gain K for mean update
        # 3. Mean: x_bar_a = x_bar_f + K @ (y - H x_bar_f)
        # 4. Reduced gain K_tilde for perturbation update
        # 5. Perturbations: X'_a = X'_f - K_tilde @ H @ X'_f
        # 6. Reassemble: X_a = x_bar_a + X'_a
        ...
```

**Ref:** Whitaker, J. S. & Hamill, T. M. (2002). *Ensemble data assimilation
without perturbed observations.* Mon. Wea. Rev., 130, 1913--1924.

---

### Gap 5: EnSRF_Serial — Serial Ensemble Square Root Filter

**Method family:** Deterministic (serial observation processing).

**Math:** Process observations one at a time. For the $k$-th scalar observation
$y_k$ with variance $R_{kk}$:

$$K_k = \frac{C^{x H_k}}{C^{H_k H_k} + R_{kk}}$$

where $C^{x H_k} \in \mathbb{R}^{N_x}$ is the cross-covariance with the $k$-th
observation, and $C^{H_k H_k} \in \mathbb{R}$ is scalar. No matrix inversion is
needed.

**Mean update** (serial):

$$\bar{x}_a^{(k)} = \bar{x}_a^{(k-1)} + K_k \bigl(y_k - H_k \bar{x}_a^{(k-1)}\bigr)$$

**Perturbation update** (serial, reduced gain):

$$\alpha_k = \frac{1}{1 + \sqrt{R_{kk} / (C^{H_k H_k} + R_{kk})}}$$

$$X'^{(k)}_a = X'^{(k-1)}_a - \alpha_k K_k H_k X'^{(k-1)}_a$$

Each observation is assimilated sequentially; after processing all $N_y$
observations, the ensemble is fully updated.

**Complexity:** $O(N_e N_x N_y)$ total. Each of the $N_y$ scalar updates costs
$O(N_e N_x)$. Avoids all matrix inversions. Requires $R$ diagonal (uncorrelated
observation errors) or a decorrelation pre-transform.

```python
class EnSRF_Serial(AbstractSequentialFilter):
    """Serial Ensemble Square Root Filter (Whitaker & Hamill 2002)."""

    def analysis(
        self,
        particles: Float[Array, "N_e N_x"],
        obs: Float[Array, "N_y"],
        obs_op: AbstractObsOperator,
        obs_noise: AbstractLinearOperator,
    ) -> AnalysisResult:
        # For k = 1, ..., N_y:
        #   1. Scalar obs variance: R_kk
        #   2. Cross-cov: C_xHk = X' @ (H_k X')^T / (N_e - 1)
        #   3. Obs variance: C_HkHk = var(H_k X')
        #   4. Gain: K_k = C_xHk / (C_HkHk + R_kk)
        #   5. Mean update: x_bar += K_k * (y_k - H_k x_bar)
        #   6. Alpha: alpha_k = 1 / (1 + sqrt(R_kk / (C_HkHk + R_kk)))
        #   7. Pert update: X' -= alpha_k * K_k * H_k X'
        ...
```

**Ref:** Whitaker, J. S. & Hamill, T. M. (2002). *Ensemble data assimilation
without perturbed observations.* Mon. Wea. Rev., 130, 1913--1924.

---

### Gap 6: ESTKF — Error Subspace Transform Kalman Filter

**Method family:** Deterministic (error-subspace eigendecomposition).

**Math:** Hybrid of ETKF and EnSRF. Project into the $(N_e - 1)$-dimensional
error subspace using a mean-preserving projection $L \in \mathbb{R}^{N_e \times (N_e - 1)}$
satisfying $L^{\!\top} L = I$ and $L^{\!\top} \mathbf{1} = 0$.

Define the reduced anomalies:

$$\tilde{Y} = Y' L \in \mathbb{R}^{N_y \times (N_e - 1)}$$

Eigendecompose the innovation precision in the reduced subspace:

$$A = (N_e - 1)\,I + \tilde{Y}^{\!\top} R^{-1} \tilde{Y} \in \mathbb{R}^{(N_e-1) \times (N_e-1)}$$

$$A = U_A \Lambda_A U_A^{\!\top}$$

The analysis weights and transform are:

$$\bar{w} = U_A \Lambda_A^{-1} U_A^{\!\top} \tilde{Y}^{\!\top} R^{-1} (y - \bar{y}_f)$$

$$W = U_A \Lambda_A^{-1/2} U_A^{\!\top} \sqrt{N_e - 1}$$

$$X_a = \bar{x}_f \mathbf{1}^{\!\top} + X'_f L \bigl(\bar{w} \mathbf{1}^{\!\top} + W\bigr)$$

**Complexity:** $O(N_e^2 N_y + N_e^3)$. Same order as ETKF but operates in the
$(N_e-1)$-dimensional subspace, giving a modest constant factor improvement.

```python
class ESTKF(AbstractSequentialFilter):
    """Error Subspace Transform Kalman Filter (Nerger et al. 2012)."""

    def analysis(
        self,
        particles: Float[Array, "N_e N_x"],
        obs: Float[Array, "N_y"],
        obs_op: AbstractObsOperator,
        obs_noise: AbstractLinearOperator,
    ) -> AnalysisResult:
        # 1. Construct mean-preserving projection L
        # 2. Reduced anomalies: Y_tilde = Y' @ L
        # 3. A = (N_e - 1) I + Y_tilde^T R^{-1} Y_tilde
        # 4. Eigendecompose A = U_A Lambda_A U_A^T
        # 5. w_bar = U_A Lambda_A^{-1} U_A^T Y_tilde^T R^{-1} (y - y_bar)
        # 6. W = U_A diag(sqrt((N_e-1)/lambda)) U_A^T
        # 7. X_a = x_bar 1^T + X'_f L (w_bar 1^T + W)
        ...
```

**Ref:** Nerger, L., Janjic, T., Schroter, J., & Hiller, W. (2012). *A unification
of ensemble square root Kalman filters.* Mon. Wea. Rev., 140, 2335--2345.

---

### Gap 7: LETKF — Local Ensemble Transform Kalman Filter

**Method family:** Deterministic (localized ETKF).

**Math:** For each grid point $i$ (or local analysis domain):

1. **Select local observations:** Find indices $\mathcal{I}_i = \{k : d(x_i, y_k) \leq r\}$ where $r$ is the localization radius.

2. **Apply distance-dependent tapering** to scale observation weights:

$$\rho_k = \rho_{\text{GC}}\!\bigl(d(x_i, y_k) / r\bigr)$$

where $\rho_{\text{GC}}$ is the Gaspari-Cohn function. Form the localized observation error covariance:

$$R_i^{\text{loc}} = \text{diag}\bigl(R_{kk} / \rho_k\bigr)_{k \in \mathcal{I}_i}$$

3. **Run ETKF locally** with the selected observations and $R_i^{\text{loc}}$:

$$\tilde{C}_i = (N_e - 1)\,I + Y'_{\mathcal{I}_i}^{\!\top} (R_i^{\text{loc}})^{-1} Y'_{\mathcal{I}_i}$$

$$T_i = \tilde{C}_i^{-1}, \qquad \bar{w}_i = T_i Y'_{\mathcal{I}_i}^{\!\top} (R_i^{\text{loc}})^{-1} d_i$$

$$W_i = \sqrt{(N_e - 1) T_i}$$

4. **Update grid point $i$:**

$$x_{a,i}^{(j)} = \bar{x}_{f,i} + X'_{f,i} (\bar{w}_i + W_i)_j$$

5. **Reassemble** all grid points into the global analysis ensemble.

The local analyses are **embarrassingly parallel** — each grid point is
independent. This is the standard algorithm for operational numerical weather
prediction (NWP).

**Complexity:** $O(N_{\text{grid}} \times (N_e^2 N_{y,\text{local}} + N_e^3))$.
Each local analysis is cheap because $N_{y,\text{local}} \ll N_y$. Total cost
scales linearly with grid size.

```python
class LETKF(AbstractSequentialFilter):
    """Local Ensemble Transform Kalman Filter (Hunt et al. 2007)."""
    localizer: AbstractLocalizer

    def analysis(
        self,
        particles: Float[Array, "N_e N_x"],
        obs: Float[Array, "N_y"],
        obs_op: AbstractObsOperator,
        obs_noise: AbstractLinearOperator,
        *,
        state_coords: Float[Array, "N_x D"],
        obs_coords: Float[Array, "N_y D"],
    ) -> AnalysisResult:
        # 1. Compute Y' = H(X_f) anomalies (once, global)
        # 2. For each grid point i (vmap):
        #    a. Select local obs indices within radius
        #    b. Extract local Y'_local, R_local (with tapering)
        #    c. Run ETKF on local subset
        #    d. Update x_a[i] for all ensemble members
        # 3. Stack local updates into global analysis
        ...
```

**Ref:** Hunt, B. R., Kostelich, E. J., & Szunyogh, I. (2007). *Efficient data
assimilation for spatiotemporal chaos: A local ensemble transform Kalman filter.*
Physica D, 230, 112--126.

---

### Gap 8: SquareRootKF — Parametric Square-Root Kalman Filter

**Method family:** Parametric (not ensemble-based). Included for reference.

**Math:** Propagates the Cholesky factor $S$ where $P = S S^{\!\top}$, guaranteeing
positive semi-definiteness by construction. The standard Kalman filter updates
$P$ directly, which can lose PSD structure due to floating-point errors.

**Forecast:**

$$S_f = \text{chol}\!\bigl(\Phi S_a S_a^{\!\top} \Phi^{\!\top} + Q\bigr)$$

or via the square-root form using QR factorization:

$$\begin{bmatrix} S_a \Phi^{\!\top} \\ \sqrt{Q} \end{bmatrix} = Q_{\text{QR}} R_{\text{QR}}, \qquad S_f = R_{\text{QR}}^{\!\top}$$

**Analysis (Potter form):**

Process observations serially. For scalar observation $y_k$ with variance $R_{kk}$:

$$v_k = S_f^{\!\top} H_k^{\!\top}$$

$$\sigma_k^2 = v_k^{\!\top} v_k + R_{kk}$$

$$K_k = S_f v_k / \sigma_k^2$$

$$\bar{x}_a = \bar{x}_f + K_k (y_k - H_k \bar{x}_f)$$

$$S_a = S_f - \frac{K_k v_k^{\!\top}}{1 + \sqrt{R_{kk}/\sigma_k^2}}$$

**Complexity:** $O(N_x^2 N_y)$ for serial processing, $O(N_x^3)$ for the QR
forecast step. Scales with state dimension, not ensemble size.

```python
class SquareRootKF(eqx.Module):
    """Parametric Square-Root Kalman Filter.

    Propagates Cholesky factor S (P = S S^T) for guaranteed PSD.
    Not ensemble-based — included as a reference implementation.
    """

    def predict(
        self,
        mean: Float[Array, "N_x"],
        chol: Float[Array, "N_x N_x"],
        dynamics_jacobian: Float[Array, "N_x N_x"],
        process_noise: Float[Array, "N_x N_x"],
    ) -> tuple[Float[Array, "N_x"], Float[Array, "N_x N_x"]]:
        # QR-based Cholesky propagation
        ...

    def update(
        self,
        mean: Float[Array, "N_x"],
        chol: Float[Array, "N_x N_x"],
        obs: Float[Array, "N_y"],
        obs_matrix: Float[Array, "N_y N_x"],
        obs_noise: Float[Array, "N_y N_y"],
    ) -> tuple[Float[Array, "N_x"], Float[Array, "N_x N_x"]]:
        # Potter serial update on Cholesky factor
        ...
```

**Ref:** Maybeck, P. S. (1979). *Stochastic Models, Estimation, and Control.*
Vol. 1, Academic Press. Chapter 7.

---

## 4  Shared Infrastructure

All ensemble filters share building blocks from `gaussx` and the filter framework:

| Component | Source | Role |
|---|---|---|
| `ensemble_mean` | `filterx._primitives` | $\bar{x} = \frac{1}{N_e}\sum_j x^{(j)}$ |
| `ensemble_anomalies` | `filterx._primitives` | $X' = (X - \bar{x}\mathbf{1}^{\!\top}) / \sqrt{N_e - 1}$ |
| `ensemble_covariance` | `filterx._primitives` | $C = X' X'^{\!\top}$ (via anomalies) |
| `ensemble_cross_covariance` | `filterx._primitives` | $C^{xH} = X' Y'^{\!\top}$ |
| `kalman_gain` | `filterx._primitives` | $K = C^{xH}(C^{HH} + R)^{-1}$ via `gaussx.solve` |
| `gaussx.solve` | `gaussx._primitives` | Structured linear solve (Woodbury, diagonal, dense) |
| `gaussx.logdet` | `gaussx._primitives` | Log-determinant for likelihood computation |
| `gaussx.cholesky` | `gaussx._primitives` | Cholesky factorization for square-root methods |
| `gaussx.Woodbury` | `gaussx._operators` | Low-rank + diagonal solve for $(C^{HH} + R)^{-1}$ |
| `AbstractLocalizer` | `filterx._components` | Gaspari-Cohn, Gaussian, cutoff tapering |
| `AbstractInflator` | `filterx._components` | Multiplicative, RTPS, RTPP inflation |

---

## 5  Comparison Table

| Filter | Family | Perturbed obs? | Localization | Serial obs? | Complexity (per step) | Notes |
|--------|--------|:-:|:-:|:-:|---|---|
| **StochasticEnKF** | Stochastic | Yes | Covariance loc. | No | $O(N_e^2 N_y + N_e^3)$ | Simplest; sampling noise |
| **ETKF** | Deterministic | No | No (global) | No | $O(N_e^2 N_y + N_e^3)$ | Symmetric sqrt |
| **ETKF_Livings** | Deterministic | No | No (global) | No | $O(N_e^2 N_y + N_e^3)$ | Random rotation breaks degeneracy |
| **EnSRF** | Deterministic | No | Covariance loc. | No | $O(N_e^2 N_y + N_y^3)$ | Separate mean/pert updates |
| **EnSRF_Serial** | Deterministic | No | Covariance loc. | Yes | $O(N_e N_x N_y)$ | No matrix inversion; requires diag $R$ |
| **ESTKF** | Deterministic | No | No (global) | No | $O(N_e^2 N_y + N_e^3)$ | Error-subspace projection |
| **LETKF** | Deterministic | No | Domain loc. | No | $O(N_{\text{grid}} N_e^2 N_{y,\text{loc}})$ | Operational NWP standard; parallel |
| **SquareRootKF** | Parametric | No | N/A | Yes | $O(N_x^2 N_y + N_x^3)$ | Not ensemble; Cholesky propagation |

**Key distinctions:**

- **Stochastic vs. deterministic:** StochasticEnKF is the only filter that
  perturbs observations. All others produce the exact Kalman analysis covariance
  (up to the low-rank ensemble approximation).
- **Localization strategy:** LETKF uses *domain localization* (local analysis
  domains). StochasticEnKF, EnSRF, and EnSRF_Serial support *covariance
  localization* (Schur product with a taper matrix). ETKF/ESTKF do not naturally
  support localization without modification.
- **Serial observation processing:** EnSRF_Serial and SquareRootKF process
  observations one at a time, avoiding matrix inversions but requiring diagonal
  (or pre-decorrelated) $R$.

---

## 6  References

1. Evensen, G. (1994). *Sequential data assimilation with a nonlinear quasi-geostrophic model using Monte Carlo methods to forecast error statistics.* J. Geophys. Res., 99(C5), 10143--10162.
2. Bishop, C. H., Etherton, B. J., & Majumdar, S. J. (2001). *Adaptive sampling with the ensemble transform Kalman filter. Part I: Theoretical aspects.* Mon. Wea. Rev., 129, 420--436.
3. Livings, D. M., Dance, S. L., & Nichols, N. K. (2008). *Unbiased ensemble square root filters.* Physica D, 237(8), 1021--1028.
4. Whitaker, J. S. & Hamill, T. M. (2002). *Ensemble data assimilation without perturbed observations.* Mon. Wea. Rev., 130, 1913--1924.
5. Nerger, L., Janjic, T., Schroter, J., & Hiller, W. (2012). *A unification of ensemble square root Kalman filters.* Mon. Wea. Rev., 140, 2335--2345.
6. Hunt, B. R., Kostelich, E. J., & Szunyogh, I. (2007). *Efficient data assimilation for spatiotemporal chaos: A local ensemble transform Kalman filter.* Physica D, 230, 112--126.
7. Maybeck, P. S. (1979). *Stochastic Models, Estimation, and Control.* Vol. 1, Academic Press.
8. Anderson, J. L. (2001). *An ensemble adjustment Kalman filter for data assimilation.* Mon. Wea. Rev., 129, 2884--2903.
9. Tippett, M. K., Anderson, J. L., Bishop, C. H., Hamill, T. M., & Whitaker, J. S. (2003). *Ensemble square root filters.* Mon. Wea. Rev., 131, 1485--1490.
10. Gaspari, G. & Cohn, S. E. (1999). *Construction of correlation functions in two and three dimensions.* Q. J. R. Meteorol. Soc., 125, 723--757.
