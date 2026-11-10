---
status: draft
version: 0.1.0
---

# ekalmX — Layer 0: Primitives

Pure functions. Stateless, differentiable, composable. No classes, no protocols — just `(Array, ...) → Array`.

All functions are compatible with `jax.jit`, `jax.grad`, and `eqx.filter_vmap`.

For each function, we show: (1) the continuous mathematical definition, (2) the numerical computation, and (3) the complexity. The math follows standard ensemble Kalman filter notation (Evensen 1994, Vetra-Carvalho et al. 2018).

> **Note**: Function signatures below are interface sketches using [jaxtyping](https://github.com/patrick-kidger/jaxtyping) annotations (`Float[Array, ...]`), [lineax](https://github.com/patrick-kidger/lineax) (`AbstractLinearOperator`), and [gaussx](../README.md) (`gaussx.operators.LowRankUpdate`). They are not runnable as-is.

---

## statistics

Ensemble statistics computed on demand from particles. Given an ensemble $\{x^{(j)}\}_{j=1}^{N_e}$ where each $x^{(j)} \in \mathbb{R}^{N_x}$:

### `ensemble_mean`

**Mathematical definition:**

$$\bar{x} = \frac{1}{N_e} \sum_{j=1}^{N_e} x^{(j)}$$

**Complexity:** $O(N_e \cdot N_x)$ — single reduction over ensemble axis.

```python
def ensemble_mean(
    particles: Float[Array, "N_e N_x"],
) -> Float[Array, "N_x"]:
    """Ensemble mean: x̄ = (1/N_e) Σⱼ xⱼ"""
```

### `ensemble_anomalies`

**Mathematical definition:**

$$X' = X - \bar{x}, \qquad X'_{j} = x^{(j)} - \bar{x}$$

The anomaly matrix $X' \in \mathbb{R}^{N_e \times N_x}$ has rows that sum to zero. It encodes the ensemble's deviation from the mean and is the fundamental building block for covariance and gain computations.

**Complexity:** $O(N_e \cdot N_x)$.

```python
def ensemble_anomalies(
    particles: Float[Array, "N_e N_x"],
) -> Float[Array, "N_e N_x"]:
    """Centered perturbations: X' = X - x̄

    Each row is xⱼ - x̄. Used as input to covariance and gain computations.
    """
```

### `ensemble_covariance`

**Mathematical definition:**

$$P = \frac{1}{N_e - 1} X'^T X' = \frac{1}{N_e - 1} \sum_{j=1}^{N_e} (x^{(j)} - \bar{x})(x^{(j)} - \bar{x})^T$$

**Numerical method:** Rather than forming the dense $N_x \times N_x$ matrix (which may be $10^4 \times 10^4$ or larger), we return a gaussx `LowRankUpdate` of rank $\leq N_e - 1$. The factored form $P = \frac{1}{N_e-1} X'^T X'$ with $X' \in \mathbb{R}^{N_e \times N_x}$ is stored as a low-rank operator, enabling Woodbury-based solves and matrix-determinant-lemma logdets via gaussx.

**Complexity:** $O(N_e \cdot N_x)$ to construct. Downstream operations (solve, logdet) exploit the rank-$(N_e-1)$ structure.

```python
def ensemble_covariance(
    particles: Float[Array, "N_e N_x"],
) -> gaussx.operators.LowRankUpdate:
    """Sample covariance as a low-rank operator.

    P = (1/(N_e - 1)) X'ᵀ X'

    Returns a gaussx LowRankUpdate of rank ≤ N_e - 1.
    Never forms the dense (N_x, N_x) matrix.

    Delegates to gaussx.recipes.ensemble_covariance.
    """
```

### `cross_covariance`

**Mathematical definition:**

$$C^{xH} = \frac{1}{N_e - 1} \sum_{j=1}^{N_e} (x^{(j)} - \bar{x})(Hx^{(j)} - \overline{Hx})^T = \frac{1}{N_e - 1} X'^T (HX)'$$

In the linear case $H x = Hx$, this reduces to $C^{xH} = P H^T$. For nonlinear observation operators, the ensemble cross-covariance provides a stochastic linearisation — an implicit, derivative-free approximation to the Jacobian $\nabla H$.

**Complexity:** $O(N_e \cdot N_x \cdot N_y)$. Returns a dense $(N_x, N_y)$ array since $N_y$ is typically small.

```python
def cross_covariance(
    particles: Float[Array, "N_e N_x"],
    obs_particles: Float[Array, "N_e N_y"],
) -> Float[Array, "N_x N_y"]:
    """Cross-covariance between state and observation space.

    C^{xH} = (1/(N_e - 1)) X'ᵀ (HX)'

    Returns a dense (N_x, N_y) array (not an operator) since
    N_y is typically small enough for dense representation.
    """
```

---

## gain

Kalman gain computation, delegating to gaussx for structured solves.

### `kalman_gain`

**Mathematical definition:**

The Kalman gain maps innovations (observation-space residuals) back to state-space corrections:

$$K = C^{xH} \left(C^{HH} + R\right)^{-1}$$

where $C^{HH} = \frac{1}{N_e-1}(HX)'^T(HX)' \in \mathbb{R}^{N_y \times N_y}$ is the ensemble covariance in observation space and $R$ is the observation noise covariance.

**Numerical method:** The innovation covariance $S = C^{HH} + R$ is a `LowRankUpdate` when $N_e \ll N_y$ (low-rank ensemble contribution + diagonal/structured noise). The solve $S^{-1}$ exploits this via the Woodbury identity through gaussx:

$$S^{-1} = (R + C^{HH})^{-1} = R^{-1} - R^{-1} (HX)' \left((N_e - 1)I + (HX)'^T R^{-1} (HX)'\right)^{-1} (HX)'^T R^{-1}$$

The inner matrix is only $N_e \times N_e$, so the total cost is $O(N_e^2 N_y + N_e^3)$ instead of $O(N_y^3)$.

**Complexity:** $O(N_e^2 N_y + N_e^3)$ via Woodbury; $O(N_y^3)$ dense fallback.

```python
def kalman_gain(
    particles: Float[Array, "N_e N_x"],
    obs_particles: Float[Array, "N_e N_y"],
    obs_noise: AbstractLinearOperator,
    solver: Optional[gaussx.SolverStrategy] = None,
) -> Float[Array, "N_x N_y"]:
    """Kalman gain: K = C^{xH} (C^{HH} + R)⁻¹

    The innovation covariance S = C^{HH} + R is a LowRankUpdate
    when N_e << N_y, enabling Woodbury solve via gaussx.

    Parameters
    ----------
    particles : (N_e, N_x)
        Prior ensemble in state space.
    obs_particles : (N_e, N_y)
        Prior ensemble mapped to observation space (H @ particles).
    obs_noise : AbstractLinearOperator
        Observation error covariance R.
    solver : optional
        gaussx solver strategy. Default: AutoSolver.

    Returns
    -------
    K : (N_x, N_y)
        Kalman gain matrix.
    """
```

---

## likelihood

Log-likelihood and innovation diagnostics for differentiable training.

### `log_likelihood`

**Mathematical definition:**

The Gaussian log-likelihood of the innovation vector $v_t = y_t - H\bar{x}_t$ under the innovation covariance $S_t = HP_tH^T + R$:

$$\log p(y_t \mid \text{forecast}) = -\frac{1}{2}\left[N_y \log(2\pi) + \log|S_t| + v_t^T S_t^{-1} v_t\right]$$

This is the training signal for differentiable data assimilation — backpropagating through $\log p$ with respect to dynamics parameters enables end-to-end learning (Decision D9).

**Numerical method:** Both `logdet(S)` and `solve(S, v)` are delegated to gaussx. When $S$ is a `LowRankUpdate` (the typical case since $P$ is rank-$(N_e-1)$), the matrix determinant lemma and Woodbury identity apply.

```python
def log_likelihood(
    innovation: Float[Array, "N_y"],
    innovation_cov: AbstractLinearOperator,
    solver: Optional[gaussx.SolverStrategy] = None,
) -> Scalar:
    """Gaussian log-likelihood of the innovation vector.

    log p(y | forecast) = -½ [N_y log(2π) + log|S| + vᵀ S⁻¹ v]

    where v = y - H x̄ (innovation) and S = H P Hᵀ + R (innovation covariance).

    Uses gaussx.ops.logdet and gaussx.ops.solve for structure-exploiting
    computation when S is a LowRankUpdate.
    """
```

### `innovation_statistics`

```python
def innovation_statistics(
    particles: Float[Array, "N_e N_x"],
    obs: Float[Array, "N_y"],
    obs_op: Callable,
    obs_noise: AbstractLinearOperator,
) -> dict:
    """Compute innovation diagnostics.

    Returns
    -------
    dict with keys:
        innovation : (N_y,) — v = y - H x̄
        innovation_cov : AbstractLinearOperator — S = H P Hᵀ + R
        normalized_innovation : (N_y,) — S^{-1/2} v
        log_likelihood : Scalar
    """
```

---

## localization

Covariance localization suppresses spurious long-range correlations in the ensemble covariance that arise from finite ensemble size. The localized covariance is formed via Schur (element-wise) product:

$$P_{\text{loc}} = \rho \circ P$$

where $\rho_{ij} = \rho(d_{ij} / r)$ is a compactly-supported correlation function of the distance $d_{ij}$ between grid points $i$ and $j$, and $r$ is the localization radius.

### `gaspari_cohn`

**Mathematical definition (Gaspari & Cohn 1999):**

A 5th-order piecewise polynomial on $z = d/r \in [0, 2]$:

$$\rho(z) = \begin{cases} -\frac{1}{4}z^5 + \frac{1}{2}z^4 + \frac{5}{8}z^3 - \frac{5}{3}z^2 + 1, & 0 \leq z \leq 1 \\ \frac{1}{12}z^5 - \frac{1}{2}z^4 + \frac{5}{8}z^3 + \frac{5}{3}z^2 - 5z + 4 - \frac{2}{3z}, & 1 < z \leq 2 \\ 0, & z > 2 \end{cases}$$

Properties: compactly supported (zero beyond $2r$), $C^2$ smooth at all transition points, positive definite (the localized covariance is guaranteed PSD).

```python
def gaspari_cohn(
    distances: Float[Array, "..."],
    radius: float,
) -> Float[Array, "..."]:
    """Gaspari-Cohn 5th-order piecewise polynomial taper.

    Compactly supported: zero beyond 2 × radius.
    Smooth (C²) at all transition points.

    Parameters
    ----------
    distances : array of pairwise distances
    radius : localization half-width (compact support at 2 × radius)

    Returns
    -------
    Taper weights in [0, 1]. Same shape as distances.
    """
```

### `gaussian_taper`

```python
def gaussian_taper(
    distances: Float[Array, "..."],
    radius: float,
) -> Float[Array, "..."]:
    """Gaussian taper: exp(-d² / (2 radius²))

    Not compactly supported — decays but never reaches zero.
    Simpler than Gaspari-Cohn but introduces weak long-range correlations.
    """
```

### `hard_cutoff`

```python
def hard_cutoff(
    distances: Float[Array, "..."],
    radius: float,
) -> Float[Array, "..."]:
    """Binary cutoff: 1 if d ≤ radius, 0 otherwise.

    Discontinuous — can cause filter instability.
    Use Gaspari-Cohn or Gaussian for production.
    """
```

### `localize`

```python
def localize(
    cov: Float[Array, "N_x N_y"],
    taper: Float[Array, "N_x N_y"],
) -> Float[Array, "N_x N_y"]:
    """Apply localization via Hadamard (element-wise) product.

    P_loc = P ⊙ L

    Works on both dense arrays and structured operators.
    """
```

---

## inflation

Ensemble inflation counteracts the systematic underestimation of uncertainty that occurs with finite ensembles. Without inflation, the ensemble spread collapses over repeated assimilation cycles, leading to filter divergence.

### `inflate_multiplicative`

**Mathematical definition:**

$$x^{(j)}_{\text{inflated}} = \bar{x} + \lambda \cdot (x^{(j)} - \bar{x}), \qquad \lambda > 1$$

Equivalently, $X'_{\text{inflated}} = \lambda X'$. The inflated covariance is $P_{\text{inflated}} = \lambda^2 P$.

```python
def inflate_multiplicative(
    particles: Float[Array, "N_e N_x"],
    factor: float,
) -> Float[Array, "N_e N_x"]:
    """Multiplicative inflation: X'_inflated = factor × X'

    Scales anomalies around the ensemble mean.
    factor > 1 increases spread, factor = 1 is identity.
    """
```

### `inflate_rtps`

**Mathematical definition (Whitaker & Hamill 2012):**

Relaxation to Prior Spread rescales the analysis anomalies so that the per-variable spread interpolates between analysis and forecast:

$$\sigma_{\text{relaxed},i} = (1 - \alpha)\,\sigma^a_i + \alpha\,\sigma^f_i$$

$$x^{(j)}_{\text{relaxed}} = \bar{x}^a + \frac{\sigma_{\text{relaxed},i}}{\sigma^a_i} \cdot (x^{(j)}_a - \bar{x}^a)$$

where $\sigma^a_i, \sigma^f_i$ are the per-variable analysis and forecast ensemble standard deviations. With $\alpha = 0$ the analysis is unchanged; with $\alpha = 1$ the forecast spread is fully restored.

```python
def inflate_rtps(
    analysis_particles: Float[Array, "N_e N_x"],
    forecast_particles: Float[Array, "N_e N_x"],
    alpha: float,
) -> Float[Array, "N_e N_x"]:
    """Relaxation to Prior Spread (Whitaker & Hamill 2012).

    σ_relaxed = (1 - α) σ_analysis + α σ_forecast

    Anomalies are rescaled so the analysis spread interpolates
    between the pure analysis spread and the forecast spread.

    Parameters
    ----------
    alpha : relaxation coefficient in [0, 1].
        0 = pure analysis, 1 = restore full forecast spread.
    """
```

### `inflate_rtpp`

```python
def inflate_rtpp(
    analysis_particles: Float[Array, "N_e N_x"],
    forecast_particles: Float[Array, "N_e N_x"],
    alpha: float,
) -> Float[Array, "N_e N_x"]:
    """Relaxation to Prior Perturbations (Zhang et al. 2004).

    X'_relaxed = (1 - α) X'_analysis + α X'_forecast

    Directly interpolates perturbations (not spread).
    """
```

### `inflate_adaptive`

```python
def inflate_adaptive(
    particles: Float[Array, "N_e N_x"],
    obs: Float[Array, "N_y"],
    obs_op: Callable,
    obs_noise: AbstractLinearOperator,
    min_factor: float = 1.0,
    max_factor: float = 1.2,
) -> Float[Array, "N_e N_x"]:
    """Adaptive multiplicative inflation (Anderson 2007).

    Estimates optimal inflation from innovation statistics.
    Clipped to [min_factor, max_factor] for stability.
    """
```

---

## perturbations

Random perturbation generation for stochastic methods.

### `perturbed_observations`

```python
def perturbed_observations(
    key: PRNGKey,
    obs: Float[Array, "N_y"],
    obs_noise: AbstractLinearOperator,
    n_ensemble: int,
) -> Float[Array, "N_e N_y"]:
    """Generate perturbed observations for stochastic EnKF.

    y_pert[j] = y + ε[j],  ε[j] ~ N(0, R)

    Uses gaussx AbstractNoise.sample or direct Cholesky sampling.
    Deterministic given the PRNG key.
    """
```

---

## patches

Array-level spatial decomposition for large domains.

### `create_patches`

```python
def create_patches(
    field: Float[Array, "N_e *spatial"],
    patch_size: tuple[int, ...],
    stride: tuple[int, ...],
) -> tuple[Float[Array, "N_patches N_e *patch_spatial"], PatchMetadata]:
    """Decompose spatial field into overlapping patches.

    overlap = patch_size - stride per dimension.

    Parameters
    ----------
    field : ensemble field with spatial dimensions
    patch_size : size of each patch per spatial dimension
    stride : step between patch origins (stride < patch_size for overlap)

    Returns
    -------
    patches : stacked patch array
    metadata : PatchMetadata for reconstruction (origins, grid info)
    """
```

### `assign_obs_to_patches`

```python
def assign_obs_to_patches(
    obs_coords: Float[Array, "N_y D"],
    obs_values: Float[Array, "N_y"],
    metadata: PatchMetadata,
    buffer: int = 0,
) -> dict[int, tuple[Array, Array]]:
    """Map observations to their enclosing patches.

    Each observation is assigned to all patches whose (extended) footprint
    contains it. Buffer extends the patch footprint by buffer grid points
    to include nearby observations.

    Returns dict: patch_id → (local_obs_values, local_obs_coords)
    """
```

### `blend_patches`

```python
def blend_patches(
    patches: Float[Array, "N_patches N_e *patch_spatial"],
    metadata: PatchMetadata,
    taper_fn: Callable = gaspari_cohn,
    taper_radius: Optional[float] = None,
) -> Float[Array, "N_e *spatial"]:
    """Blend overlapping patches back into a global field.

    Uses distance-weighted blending in overlap regions:
    x_merged = Σ w(d) x_patch / Σ w(d)

    where w(d) is the taper function applied to distance from patch center.
    """
```
