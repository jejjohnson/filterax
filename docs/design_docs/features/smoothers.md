---
status: draft
version: 0.1.0
---

# ekalmX — Ensemble Smoothers

**Subject:** Backward-pass methods that refine filter estimates using future
observations. Smoothers provide better state estimates than filters alone by
incorporating all observations (past + future).

**Date:** 2026-04-03

---

## 1  Scope

Sequential ensemble filters produce the filtering distribution
$p(x_t \mid y_{1:t})$ — the best estimate given observations *up to* time $t$.
Ensemble smoothers refine these estimates into the smoothing distribution
$p(x_t \mid y_{1:T})$ for $T > t$, incorporating future observations via a
backward pass over stored filter results.

**In scope:** Backward-pass ensemble smoothers that operate on the output of
any `AbstractSequentialFilter` (EnKS, EnsembleRTS, fixed-lag, square-root,
iterative).

**Out of scope:** Forward-only filters, EKP processes (iterative parameter
estimation), variational smoothers (4D-Var).

**Key insight:** All ensemble smoothers share the same backward-pass structure
as the classical Rauch-Tung-Striebel smoother — only the gain computation and
storage strategy differ. The smoother gain is always a cross-covariance
times an inverse forecast covariance, estimated from the ensemble.

---

## 2  Common Mathematical Framework

The filter produces the filtering distribution at each time step:

$$p(x_t \mid y_{1:t}) \approx \{x_t^{a,(j)}\}_{j=1}^{N_e}$$

The smoother refines this to the smoothing distribution using a backward
recursion from $t = T-1$ down to $t = 0$:

$$G_t = C^{a,f}_{t,t+1} \bigl(C^{f,f}_{t+1}\bigr)^{-1}$$

$$X^s_t = X^a_t + G_t \bigl(X^s_{t+1} - X^f_{t+1}\bigr)$$

where:

- $X^a_t$ — analysis (filtered) ensemble at time $t$
- $X^f_{t+1}$ — forecast ensemble at time $t+1$ (propagated from $X^a_t$)
- $X^s_{t+1}$ — smoothed ensemble at time $t+1$ (already computed in backward pass)
- $C^{a,f}_{t,t+1}$ — cross-covariance between analysis at $t$ and forecast at $t+1$
- $C^{f,f}_{t+1}$ — forecast covariance at $t+1$
- $G_t$ — smoother gain matrix

This backward pass has the same structure as the classical RTS smoother for
Kalman filters. The ensemble approximation replaces parametric covariance
matrices with sample covariances computed from the ensemble members.

---

## 3  Gap Catalog

### Gap 1: EnKS — Ensemble Kalman Smoother

**Ref:** Evensen & van Leeuwen (2000)

**Math:** Standard backward pass. At each time step $t$ from $T-1$ down to $0$:

$$G_t = C^{a,f}_{t,t+1} \bigl(C^{f,f}_{t+1}\bigr)^{-1}$$

$$X^s_t = X^a_t + G_t \bigl(X^s_{t+1} - X^f_{t+1}\bigr)$$

where the cross-covariance and forecast covariance are estimated from the
ensemble:

$$C^{a,f}_{t,t+1} = \frac{1}{N_e - 1} A_t' (F_{t+1}')^\top, \qquad C^{f,f}_{t+1} = \frac{1}{N_e - 1} F_{t+1}' (F_{t+1}')^\top$$

with $A_t' = X^a_t - \bar{x}^a_t$ and $F_{t+1}' = X^f_{t+1} - \bar{x}^f_{t+1}$ being the
ensemble anomaly matrices.

**Requires:** Storing all filter results (analysis + forecast ensembles) for
the full time window.

**Complexity:** $O(T \times N_e^2 \times N_x)$ — one gain computation per time step.

**Compatible filters:** Any `AbstractSequentialFilter` (StochasticEnKF, ETKF, EnSRF, ESTKF, LETKF).

```python
class EnKS(eqx.Module):
    """Ensemble Kalman Smoother (Evensen & van Leeuwen 2000).

    Standard backward pass over stored filter results.
    """

    def smooth(
        self,
        filter_results: list[AnalysisResult],
        forecast_particles: list[Float[Array, "N_e N_x"]],
    ) -> list[AnalysisResult]:
        ...
```

---

### Gap 2: EnsembleRTS — Ensemble Rauch-Tung-Striebel Smoother

**Ref:** Rauch, Tung & Striebel (1965); ensemble formulation following
Evensen (2003)

**Math:** Ensemble analog of the classical RTS smoother. Uses the same
backward recursion as EnKS but computes the smoother gain via an
ensemble-estimated dynamics Jacobian:

$$G_t = C^{a,a}_t M_{t+1}^\top \bigl(M_{t+1} C^{a,a}_t M_{t+1}^\top + Q_t\bigr)^{-1}$$

where $M_{t+1}$ is the linearized dynamics operator estimated from the
ensemble:

$$M_{t+1} \approx F_{t+1}' (A_t')^\dagger$$

In practice, when the dynamics are applied via `AbstractDynamics` and the
ensemble provides both analysis and forecast members, the cross-covariance
formulation reduces to the same computation as EnKS. The distinction is
conceptual: EnsembleRTS explicitly acknowledges the role of the dynamics
Jacobian, making it natural to incorporate model error covariance $Q_t$.

**Complexity:** $O(T \times N_e^2 \times N_x)$ — same as EnKS.

**Compatible filters:** Any `AbstractSequentialFilter`.

```python
class EnsembleRTS(eqx.Module):
    """Ensemble Rauch-Tung-Striebel Smoother.

    Ensemble analog of the classical RTS smoother.
    Optionally incorporates model error covariance Q.
    """

    def smooth(
        self,
        filter_results: list[AnalysisResult],
        forecast_particles: list[Float[Array, "N_e N_x"]],
        dynamics: AbstractDynamics,
    ) -> list[AnalysisResult]:
        ...
```

---

### Gap 3: FixedLagSmoother — Fixed-Lag Ensemble Smoother

**Ref:** Anderson & Anderson (1999); Nerger et al. (2012)

**Math:** Only smooths the last $L$ time steps at each new observation.
The backward pass is truncated:

$$\text{for } k = t-1, \ldots, \max(0, t - L): \quad X^s_k = X^a_k + G_k (X^s_{k+1} - X^f_{k+1})$$

At each new filter step, only $L$ backward corrections are applied.
Old states beyond the lag window are finalized and discarded from memory.

**Bounded memory:** $O(L \times N_e \times N_x)$ — independent of the total
number of time steps $T$.

**Use case:** Online / real-time applications where you cannot wait for all
data before producing smoothed estimates. Also useful when $T$ is very large
and storing all filter results is infeasible.

**Complexity:** $O(L \times N_e^2 \times N_x)$ per new observation.

**Compatible filters:** Any `AbstractSequentialFilter`.

```python
class FixedLagSmoother(eqx.Module):
    """Fixed-lag ensemble smoother.

    Only looks back a fixed number of time steps (lag).
    Bounded memory -- suitable for online applications.
    """
    lag: int = eqx.field(static=True)

    def update(
        self,
        new_filter_result: AnalysisResult,
        new_forecast: Float[Array, "N_e N_x"],
        history: list[AnalysisResult],
    ) -> list[AnalysisResult]:
        """Smooth the last `lag` time steps given new observation."""
        ...
```

---

### Gap 4: Ensemble Square Root Smoother

**Ref:** Tippett et al. (2003); Whitaker & Compo (2002)

**Math:** Deterministic backward pass — no perturbed observations at any
stage. The smoother gain is applied to the ensemble perturbation matrix
directly using a square root factorization:

$$X^{s\prime}_t = X^{a\prime}_t \, T_t^s$$

where $T_t^s$ is the backward transform matrix derived from the smoother gain
in perturbation space. This preserves the deterministic properties of the
forward square root filter (ETKF, EnSRF, ESTKF).

**Key property:** No sampling noise in either the forward or backward pass.
Ensemble mean and covariance are updated consistently.

**Complexity:** $O(T \times N_e^2 \times N_x)$ — same as EnKS but with
deterministic updates.

**Compatible filters:** ETKF, EnSRF, ESTKF (deterministic forward filters).

```python
class EnsembleSqrtSmoother(eqx.Module):
    """Ensemble Square Root Smoother.

    Deterministic backward pass -- pairs with ETKF/EnSRF forward pass.
    No perturbed observations in either direction.
    """

    def smooth(
        self,
        filter_results: list[AnalysisResult],
        forecast_particles: list[Float[Array, "N_e N_x"]],
    ) -> list[AnalysisResult]:
        ...
```

---

### Gap 5: IES — Iterative Ensemble Smoother

**Ref:** Chen & Oliver (2013); Evensen et al. (2019)

**Math:** Multiple passes over the same data window using a Gauss-Newton-style
iteration. Unlike the single-pass smoothers above, IES updates the ensemble
parameters using *all* observations simultaneously:

$$\theta_{i+1}^{(j)} = \theta_0^{(j)} + C^{\theta G}_i \bigl(C^{GG}_i + \Gamma_y\bigr)^{-1} \bigl(y + \epsilon^{(j)} - G(\theta_i^{(j)}) - \Gamma_y (\theta_i^{(j)} - \theta_0^{(j)})\bigr)$$

where $G(\theta)$ is the full forward model evaluated over the entire
assimilation window, $\Gamma_y$ is the observation noise covariance, and the
iteration index $i$ runs until convergence.

**Use case:** History matching and reservoir simulation, where the forward
model maps parameters to a full observation time series. The iterative
approach handles nonlinearity better than a single backward pass.

**Complexity:** $O(n_{\text{iter}} \times J \times N_d)$ forward model
evaluations, where $J$ is the ensemble size and $N_d$ is the total number of
observations across all time steps.

**Compatible filters:** Standalone — does not require a sequential forward
filter. Uses `AbstractDynamics` directly.

```python
class IES(eqx.Module):
    """Iterative Ensemble Smoother (Chen & Oliver 2013).

    Gauss-Newton-style iteration over a data window.
    Used for history matching / reservoir simulation.
    """
    n_iterations: int = eqx.field(static=True)
    scheduler: AbstractScheduler

    def solve(
        self,
        particles: Float[Array, "J N_p"],
        obs: Float[Array, "N_d"],
        noise_cov: AbstractLinearOperator,
        forward_model: Callable[[Float[Array, "N_p"]], Float[Array, "N_d"]],
    ) -> AnalysisResult:
        """Run iterative smoother to convergence.

        Parameters
        ----------
        particles : prior ensemble of parameters
        obs : all observations (concatenated across time)
        noise_cov : observation noise covariance
        forward_model : maps parameters -> full observation prediction

        Returns
        -------
        AnalysisResult with updated parameter ensemble
        """
        ...
```

---

## 4  Shared Infrastructure

All ensemble smoothers share infrastructure with the sequential filter layer.
They only differ in the backward-pass gain computation and storage strategy:

| Component | Source | Notes |
|---|---|---|
| Ensemble anomaly matrix | `_primitives.ensemble_anomalies` | $A' = X - \bar{x}\mathbf{1}^\top$ |
| Sample cross-covariance | `_primitives.cross_covariance` | $C^{ab} = \frac{1}{N_e-1} A' B'^\top$ |
| Sample covariance | `_primitives.sample_covariance` | $C^{ff} = \frac{1}{N_e-1} F' F'^\top$ |
| Covariance solve | `gaussx.solve` | Structural dispatch for $C^{-1} v$ |
| Filter results storage | `AnalysisResult` | PyTree — stored per time step |
| Forecast propagation | `AbstractDynamics` | Forward model interface |
| Observation operator | `AbstractObsOperator` | State-to-obs mapping |
| Noise model | `AbstractNoise` | Observation error covariance $R$ |
| Localization | `AbstractLocalizer` | Optional — suppress spurious correlations in gain |

---

## 5  Comparison Table

| Smoother | Memory | Mode | Passes | Deterministic | Compatible Filters |
|---|---|---|---|---|---|
| EnKS | $O(T \cdot N_e \cdot N_x)$ | Offline | 1 backward | No (inherits filter) | Any |
| EnsembleRTS | $O(T \cdot N_e \cdot N_x)$ | Offline | 1 backward | No (inherits filter) | Any |
| FixedLagSmoother | $O(L \cdot N_e \cdot N_x)$ | Online | Rolling | No (inherits filter) | Any |
| EnsembleSqrtSmoother | $O(T \cdot N_e \cdot N_x)$ | Offline | 1 backward | Yes | ETKF, EnSRF, ESTKF |
| IES | $O(J \cdot N_p + J \cdot N_d)$ | Offline | $n_{\text{iter}}$ | No | Standalone |

**Key trade-offs:**

- **EnKS vs EnsembleRTS:** Mathematically equivalent when model error $Q = 0$.
  EnsembleRTS provides a natural place to incorporate model error.
- **FixedLagSmoother:** Only option for real-time / streaming applications.
  Accuracy approaches full smoother as lag $L \to T$.
- **EnsembleSqrtSmoother:** Preferred when using deterministic forward filters
  to avoid introducing sampling noise in the backward pass.
- **IES:** Different paradigm — iterative, not sequential. Best for strongly
  nonlinear problems where a single backward pass is insufficient.

---

## 6  References

1. Evensen, G. & van Leeuwen, P. J. (2000). *An Ensemble Kalman Smoother for Nonlinear Dynamics.* Monthly Weather Review, 128(6), 1852-1867.
2. Rauch, H. E., Tung, F. & Striebel, C. T. (1965). *Maximum Likelihood Estimates of Linear Dynamic Systems.* AIAA Journal, 3(8), 1445-1450.
3. Chen, Y. & Oliver, D. S. (2013). *Levenberg-Marquardt Forms of the Iterative Ensemble Smoother for Efficient History Matching and Uncertainty Quantification.* Computational Geosciences, 17(4), 689-703.
4. Evensen, G., Raanes, P. N., Stordal, A. S. & Hove, J. (2019). *Efficient Implementation of an Iterative Ensemble Smoother for Data Assimilation and Reservoir History Matching.* Frontiers in Applied Mathematics and Statistics, 5, 47.
5. Tippett, M. K., Anderson, J. L., Bishop, C. H., Hamill, T. M. & Whitaker, J. S. (2003). *Ensemble Square Root Filters.* Monthly Weather Review, 131(7), 1485-1490.
6. Whitaker, J. S. & Compo, G. P. (2002). *A Comparison of Variants of the Ensemble Square Root Filter.* Monthly Weather Review, 130(7), 1913-1924.
7. Nerger, L., Janjic, T., Schroter, J. & Hiller, W. (2012). *A Regulated Localization Scheme for Ensemble-Based Kalman Filters.* Quarterly Journal of the Royal Meteorological Society, 138(664), 802-812.
