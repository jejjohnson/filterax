---
status: draft
version: 0.1.0
---

# ekalmX x Ensemble Kalman Processes

**Subject:** Iterative ensemble methods for derivative-free parameter estimation
and calibration of expensive black-box simulators.

**Date:** 2026-04-03

---

## 1  Scope

Ensemble Kalman Processes (EKP) are iterative ensemble methods that solve
inverse problems without requiring gradients of the forward model. They are
the tool of choice when the simulator $G$ is expensive, non-differentiable,
or only available as a black box (climate models, PDE solvers, lab equipment).

All methods in this catalog implement `AbstractProcess` (see
[api/components.md](../api/components.md)). The caller provides forward
evaluations $\{G(\theta^{(j)})\}$ at each iteration; the process provides the
ensemble update rule.

**In scope:** EKI, ETKI, EKS/ALDI, UKI, GNKI, SparseEKI, TEKI
**Out of scope:** Sequential data assimilation filters (`AbstractSequentialFilter`),
smoothers, neural-operator surrogates (those are upstream of $G$).

**Key insight:** Every method in this catalog shares the same outer loop --
only the update rule and state representation differ. The `AbstractProcess`
protocol (init / update) captures this cleanly: the caller owns the forward
model calls, the process owns the ensemble algebra.

---

## 2  Common Mathematical Framework

The EKP framework (Iglesias, Law & Stuart 2013) treats parameter estimation
as an iterative filtering problem in artificial time.

**Forward model.** $G: \mathbb{R}^p \to \mathbb{R}^d$ maps parameters
$\theta$ to observables.

**Observations.** $y = G(\theta^\dagger) + \eta$, where
$\eta \sim \mathcal{N}(0, \Gamma)$ is observation noise.

**Ensemble.** $\{\theta_n^{(j)}\}_{j=1}^J$, with forward evaluations
$\{G(\theta_n^{(j)})\}$.

**Empirical statistics.**

$$\bar\theta_n = \frac{1}{J}\sum_j \theta_n^{(j)}, \qquad \bar G_n = \frac{1}{J}\sum_j G(\theta_n^{(j)})$$

$$C_n^{\theta G} = \frac{1}{J-1}\sum_j (\theta_n^{(j)} - \bar\theta_n)(G(\theta_n^{(j)}) - \bar G_n)^\top \quad \text{(cross-covariance)}$$

$$C_n^{GG} = \frac{1}{J-1}\sum_j (G(\theta_n^{(j)}) - \bar G_n)(G(\theta_n^{(j)}) - \bar G_n)^\top \quad \text{(obs-space covariance)}$$

$$C_n^{\theta\theta} = \frac{1}{J-1}\sum_j (\theta_n^{(j)} - \bar\theta_n)(\theta_n^{(j)} - \bar\theta_n)^\top \quad \text{(parameter covariance)}$$

**Generic update.**

$$\theta_{n+1}^{(j)} = \theta_n^{(j)} + \Delta t_n \cdot C_n^{\theta G}\bigl(C_n^{GG} + \Delta t_n^{-1}\,\Gamma\bigr)^{-1}\bigl(y - G(\theta_n^{(j)})\bigr)$$

The step size $\Delta t_n$ plays the role of a learning rate. When
$\Delta t_n \to 0$ the update vanishes; when $\Delta t_n \to \infty$ the
noise covariance $\Gamma$ is ignored and the update becomes a full
Kalman step.

**Connection to gradient flow.** In the continuous-time limit
($\Delta t \to 0$), EKI implements preconditioned gradient descent on the
data-misfit functional $\frac{1}{2}\|y - G(\theta)\|_{\Gamma^{-1}}^2$, where
the preconditioner is the ensemble-estimated covariance $C^{\theta\theta}$.
The ensemble provides a derivative-free approximation to the Jacobian
$\nabla_\theta G$ via the cross-covariance $C^{\theta G}$.

---

## 3  Gap Catalog

### Gap 1: EKI -- Ensemble Kalman Inversion

**Ref:** Iglesias, Law & Stuart (2013)

**Purpose:** Standard inversion (optimization). Produces a point estimate
of $\theta$ by iterating the ensemble update until convergence.

**Update:**

$$\theta_{n+1}^{(j)} = \theta_n^{(j)} + \Delta t_n \cdot C_n^{\theta G}\bigl(C_n^{GG} + \Delta t_n^{-1}\,\Gamma\bigr)^{-1}\bigl(y - G(\theta_n^{(j)})\bigr)$$

**Behavior:** The ensemble collapses to a point estimate as $n \to \infty$.
The ensemble mean $\bar\theta_n$ converges to a regularized least-squares
solution. Ensemble spread vanishes -- no posterior uncertainty.

**Complexity:** $O(J^2 d + J p)$ per step ($J$ = ensemble size, $d$ = obs
dimension, $p$ = parameter dimension).

```python
class EKI(AbstractProcess):
    scheduler: AbstractScheduler
    prior_mean: Optional[Float[Array, "N_p"]] = None
    prior_cov: Optional[AbstractLinearOperator] = None
```

**When to use:** Default choice for calibration when only a point estimate
is needed. Works well even when $J \ll p$ (underdetermined regime).

---

### Gap 2: ETKI -- Ensemble Transform Kalman Inversion

**Purpose:** Output-scalable variant of EKI. Performs the update via a
transform matrix in ensemble space rather than observation space.

**Math:** Instead of forming the $d \times d$ matrix
$(C^{GG} + \Delta t^{-1}\Gamma)$, ETKI works in the $J \times J$ ensemble
subspace:

$$T = \bigl(I_J + \Delta t \cdot Y^\top \Gamma^{-1} Y\bigr)^{-1}$$

where $Y \in \mathbb{R}^{d \times J}$ is the matrix of centered forward
evaluations. The ensemble is updated via $\Theta_{n+1} = \Theta_n \cdot W$
for an appropriate weight matrix $W$ derived from $T$.

**Complexity:** $O(J^2 d)$ per step instead of $O(d^3)$. The key advantage
is avoiding the $d \times d$ solve when $d \gg J$.

```python
class ETKI(AbstractProcess):
    scheduler: AbstractScheduler
```

**When to use:** High-dimensional observation spaces ($d \gg J$), e.g.
satellite retrievals, image-based observations, full-field comparisons.

---

### Gap 3: EKS / ALDI -- Ensemble Kalman Sampler

**Ref:** Garbuno-Inigo, Hoffmann, Li & Stuart (2020)

**Purpose:** Posterior sampling. Unlike EKI, the ensemble does NOT collapse --
it produces approximate posterior samples at stationarity.

**Update (ALDI form):**

$$d\theta^{(j)} = C_n^{\theta G}\,\Gamma^{-1}(y - G(\theta^{(j)}))\,dt - \frac{d+1}{J}(\theta^{(j)} - \bar\theta)\,dt + \sqrt{2\,C_n^{\theta\theta}}\,dW^{(j)}$$

The first term drives particles toward the data. The second term is a
finite-sample drift correction. The third term is Langevin noise scaled by
the ensemble covariance, ensuring ergodicity -- the ensemble explores the
posterior rather than collapsing.

**Behavior:** Ensemble spread is maintained. At stationarity, particles are
approximate samples from the posterior $p(\theta | y)$. Requires careful
step-size control (use `EKSStableScheduler`).

```python
class EKS_Process(AbstractProcess):
    scheduler: AbstractScheduler  # use EKSStableScheduler
```

**When to use:** When posterior uncertainty quantification is needed, not
just a point estimate. Climate sensitivity studies, model structural
uncertainty, Bayesian model evidence.

---

### Gap 4: UKI -- Unscented Kalman Inversion

**Ref:** Huang, Schneider & Stuart (2022)

**Purpose:** Parametric inversion with uncertainty. Maintains an explicit
mean $\mu_n$ and covariance $\Sigma_n$ (not a random ensemble). Uses
$2p+1$ deterministic sigma points for quadrature.

**Update:**

$$\text{Sigma points:} \quad \hat\theta^0 = \hat\mu, \quad \hat\theta^{\pm j} = \hat\mu \pm c_j\,[\sqrt{\hat\Sigma}]_j, \quad j = 1,\dots,p$$

$$C^{\theta y} = \sum_j W_j\,(\hat\theta^{(j)} - \hat\mu)(\hat y^{(j)} - \hat y^0)^\top$$

$$C^{yy} = \sum_j W_j\,(\hat y^{(j)} - \hat y^0)(\hat y^{(j)} - \hat y^0)^\top + \Sigma_\nu$$

$$\mu_{n+1} = \hat\mu + C^{\theta y}(C^{yy})^{-1}(y - \hat y^0)$$

$$\Sigma_{n+1} = \hat\Sigma - C^{\theta y}(C^{yy})^{-1}(C^{\theta y})^\top$$

**Behavior:** Deterministic -- no sampling noise. Does not suffer from
ensemble collapse. The covariance $\Sigma_n$ provides calibrated
uncertainty estimates (converges to the posterior covariance in the
linear-Gaussian case).

**Complexity:** $O(p^2 d)$ per step. Requires $2p+1$ forward evaluations
per iteration (fixed, not tunable).

```python
class UKI(AbstractProcess):
    scheduler: AbstractScheduler
    alpha: float = 1.0   # sigma-point spread
    beta: float = 2.0    # higher-order moment weighting
    kappa: float = 0.0   # secondary scaling

    def init(self, mean, covariance, obs, noise_cov) -> UKIState: ...
    def update(self, state: UKIState, forward_evals) -> UKIState: ...
```

**When to use:** Moderate parameter dimension ($p \lesssim 100$) where
calibrated uncertainty is important. Better UQ than EKI, deterministic
(reproducible), no ensemble-size tuning.

---

### Gap 5: GNKI -- Gauss-Newton Kalman Inversion

**Purpose:** Faster convergence via explicit Jacobian estimation from
ensemble perturbations.

**Math:** The ensemble-estimated Jacobian is:

$$\tilde J_n \approx C_n^{\theta G}(C_n^{\theta\theta})^{-1}$$

This is used in a Gauss-Newton update:

$$\delta\theta = \bigl(\tilde J_n^\top\,\Gamma^{-1}\,\tilde J_n + \Sigma_{\text{prior}}^{-1}\bigr)^{-1}\tilde J_n^\top\,\Gamma^{-1}(y - G(\theta))$$

The update includes both a data-fit term and a prior-pull term:

$$\theta_{n+1}^{(j)} = \theta_n^{(j)} + \alpha\bigl\{K_n(y - G(\theta_n^{(j)})) + (I - K_n\tilde J_n)(m_0 - \theta_n^{(j)})\bigr\}$$

**Behavior:** Faster convergence than EKI for well-conditioned problems.
In the linear-Gaussian case, GNKI recovers the exact posterior mean and
covariance. Requires $J > p$ for $C^{\theta\theta}$ to be invertible.

```python
class GNKI(AbstractProcess):
    scheduler: AbstractScheduler
```

**When to use:** Well-conditioned inverse problems where fast convergence
matters. Requires $J > p$ (cannot operate in the underdetermined regime
unlike EKI).

---

### Gap 6: SparseEKI -- Sparse Ensemble Kalman Inversion

**Ref:** Schneider, Stuart & Wu

**Purpose:** Variable selection and sparse parameter recovery. Adds an
$L_1$ sparsity penalty to the EKI update.

**Math:** Modified update with proximal operator for $L_1$ regularization:

$$\theta_{n+1}^{(j)} = \text{prox}_{\lambda\|\cdot\|_1}\bigl(\theta_n^{(j)} + \Delta t \cdot K_n(y - G(\theta_n^{(j)}))\bigr)$$

where $\text{prox}_{\lambda\|\cdot\|_1}(z)_i = \text{sign}(z_i)\max(|z_i| - \lambda, 0)$ is the soft-thresholding operator.

**Behavior:** Drives inactive parameters to exactly zero. Useful when the
true parameter vector is sparse or when performing feature/variable
selection. The proximal step is applied element-wise after each EKI update.

```python
class SparseInversion(AbstractProcess):
    scheduler: AbstractScheduler
    penalty: str = "l1"       # "l0" | "l1"
    penalty_weight: float = 0.1
```

**When to use:** High-dimensional parameter spaces where most parameters
are expected to be zero or near-zero. Source localization, sensor
placement, sparse physics discovery.

---

### Gap 7: TEKI -- Tikhonov-Regularized Ensemble Kalman Inversion

**Ref:** Chada, Chen & Stuart

**Purpose:** Prevents ensemble collapse toward degenerate solutions by
adding prior regularization directly to the EKI update.

**Math:** The forward map, data, and noise covariance are augmented:

$$\tilde G(\theta) = \begin{bmatrix} G(\theta) \\ \theta \end{bmatrix}, \qquad \tilde y = \begin{bmatrix} y \\ m_0 \end{bmatrix}, \qquad \tilde\Gamma = \begin{bmatrix} \Gamma & 0 \\ 0 & C_0 \end{bmatrix}$$

The standard EKI update is then applied in the augmented space. The
identity block in $C^{\theta\tilde G}$ pulls particles back toward the
prior mean $m_0$, preventing the ensemble from drifting arbitrarily far
from the prior.

**Behavior:** Converges to the MAP estimate (not just MLE). More stable
than vanilla EKI when the problem is ill-posed. The regularization
strength is controlled by the prior covariance $C_0$.

```python
class TEKI(AbstractProcess):
    scheduler: AbstractScheduler
    prior_mean: Float[Array, "N_p"]
    prior_cov: AbstractLinearOperator
```

**When to use:** Ill-posed inverse problems where vanilla EKI diverges or
produces physically implausible solutions. When the MAP estimate
(incorporating prior information) is preferred over the MLE.

---

## 4  Shared Infrastructure

All EKP methods share common infrastructure -- they differ only in the
update rule and state representation.

| Component | Source | Notes |
|---|---|---|
| Ensemble statistics | `ekalmX._src.ensemble` | $\bar\theta$, $C^{\theta G}$, $C^{GG}$, $C^{\theta\theta}$ |
| `FixedScheduler` | `ekalmX.schedulers` | Constant $\Delta t$ |
| `DataMisfitController` | `ekalmX.schedulers` | Adaptive $\Delta t$; terminates when `algo_time` reaches 1.0 |
| `EKSStableScheduler` | `ekalmX.schedulers` | Stability-aware for EKS/ALDI Langevin dynamics |
| gaussx operators | `gaussx` | Structured covariance: `DiagonalLinearOperator`, `LowRankUpdate`, `Kronecker` |
| optax integration | `optax` | Learning rate schedules, gradient clipping for hybrid methods |
| `ProcessState` | `ekalmX.types` | Particles, forward evals, obs, noise_cov, step, algo_time |
| `UKIState` | `ekalmX.types` | Mean, covariance (parametric state for UKI) |

**Scheduler selection guidance:**

- `FixedScheduler(dt=1.0)` -- simple, requires manual tuning.
- `DataMisfitController(target_misfit=1.0)` -- adaptive, standard choice for EKI/ETKI/GNKI. Adjusts $\Delta t$ so the data misfit decreases at a controlled rate.
- `EKSStableScheduler(max_dt=1.0)` -- required for EKS/ALDI to prevent divergence in the Langevin dynamics.

---

## 5  Comparison Table

| Method | Convergence | Output | Obs scaling | Ensemble size | UQ | Best for |
|--------|-------------|--------|-------------|---------------|-----|----------|
| **EKI** | Collapse $\to$ point | MLE / regularized LS | $O(d^3)$ or $O(J^2 d)$ | $J \geq p+1$ (flexible) | None (spread $\to 0$) | Default calibration |
| **ETKI** | Collapse $\to$ point | MLE | $O(J^2 d)$ | $J \geq p+1$ | None | High-dim observations |
| **EKS/ALDI** | Ergodic (no collapse) | Posterior samples | $O(J^2 d)$ | $J \geq p+1$ (larger = better) | Full posterior | Uncertainty quantification |
| **UKI** | Parametric convergence | MAP + covariance | $O(p^2 d)$ | $2p+1$ (fixed) | Calibrated $\Sigma$ | Moderate $p$, needs UQ |
| **GNKI** | Fast (Gauss-Newton) | MAP + spread | $O(p^3)$ | $J > p$ (required) | Via ensemble spread | Well-conditioned, fast convergence |
| **SparseEKI** | Collapse + sparsity | Sparse point estimate | $O(J^2 d)$ | $J \geq p+1$ | None | Variable selection |
| **TEKI** | Collapse $\to$ MAP | MAP estimate | $O((d+p)^3)$ | $J \geq p+1$ | None | Ill-posed problems |

**Decision tree:**

1. Need posterior samples? $\to$ **EKS/ALDI**
2. Need calibrated covariance? $\to$ **UKI** (if $p \lesssim 100$)
3. Sparse parameters? $\to$ **SparseEKI**
4. Ill-posed / need prior regularization? $\to$ **TEKI**
5. High-dim observations ($d \gg J$)? $\to$ **ETKI**
6. Well-conditioned, want fast convergence? $\to$ **GNKI** (if $J > p$)
7. Otherwise $\to$ **EKI** (simple, robust default)

---

## 6  References

1. Iglesias, M. A., Law, K. J. H., & Stuart, A. M. (2013). *Ensemble Kalman methods for inverse problems.* Inverse Problems, 29(4), 045001.
2. Huang, D. Z., Schneider, T., & Stuart, A. M. (2022). *Iterated Kalman methodology for inverse problems.* Journal of Computational Physics, 463, 111262.
3. Garbuno-Inigo, A., Hoffmann, F., Li, W., & Stuart, A. M. (2020). *Interacting Langevin diffusions: gradient structure and ensemble Kalman sampler.* SIAM Journal on Applied Dynamical Systems, 19(1), 412-441.
4. Chada, N. K., Chen, Y., & Stuart, A. M. (2019). *Tikhonov regularization within ensemble Kalman inversion.* SIAM Journal on Numerical Analysis.
5. Kovachki, N. B., & Stuart, A. M. (2019). *Ensemble Kalman inversion: a derivative-free technique for machine learning tasks.* Inverse Problems, 35(9), 095005.
6. Schneider, T., Stuart, A. M., & Wu, J.-L. (2022). *Ensemble Kalman inversion for sparse learning of dynamical systems from time-averaged data.* Journal of Computational Physics.
7. Calvello, E., Reich, S., & Stuart, A. M. (2022). *Ensemble Kalman methods: a mean field perspective.* arXiv:2209.11371.
