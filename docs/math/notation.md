# Notation

Conventions used throughout the Mathematical Reference. The single
most important one: **the ensemble matrix stacks members as rows**,
$X \in \mathbb{R}^{N_e \times N_x}$, so axis 0 of every ensemble
array is the ensemble axis — matching the `(N_e, N_x)` shape of every
filterax `particles` argument.

## Dimensions

| Symbol | Description |
|---|---|
| $N_e$ | Ensemble size (number of members) |
| $N_x$ | State dimension |
| $N_y$ | Observation dimension (per assimilation window) |
| $N_z$ | Latent dimension (chapter 15) |
| $T$ | Number of assimilation windows / time steps |
| $J$, $N_p$, $N_d$ | Ensemble size, parameter dim, data dim for EKP processes (chapter 9) |

## Ensemble quantities

| Symbol | Description |
|---|---|
| $X \in \mathbb{R}^{N_e \times N_x}$ | Ensemble matrix; members as rows, axis 0 = ensemble |
| $x^{(j)}$ | The $j$-th ensemble member (a row of $X$) |
| $\bar{x} = \frac{1}{N_e} \sum_j x^{(j)}$ | Ensemble mean |
| $X' = X - \mathbf{1}\bar{x}^\top$ | Anomaly (perturbation) matrix |
| $P = \frac{1}{N_e - 1} X'^\top X'$ | Sample (forecast-error) covariance, Bessel-corrected |
| $Y = \mathcal{H}(X)$ | Ensemble mapped to observation space, $(N_e, N_y)$ |
| $Y'$ | Observation-space anomaly matrix |
| $C^{xH}$ | State–observation cross-covariance $\frac{1}{N_e-1} X'^\top Y'$ |
| $C^{HH}$ | Observation-space covariance $\frac{1}{N_e-1} Y'^\top Y'$ |
| $\cdot^f$, $\cdot^a$ | Forecast (prior) / analysis (posterior) superscripts |
| $\sigma_i$ | Per-variable ensemble standard deviation (spread) |
| $w_j$ | Importance weight of member $j$ |

## Observation model

| Symbol | Description |
|---|---|
| $y \in \mathbb{R}^{N_y}$ | Observation vector |
| $\mathcal{H}$ | Observation operator (possibly nonlinear); $H$ its linear(ised) matrix form |
| $R$ | Observation error covariance |
| $d = y - \mathcal{H}(\bar{x})$ | Innovation (also $v$ in some L1 docstrings) |
| $S = C^{HH} + R$ | Innovation covariance |
| $K = C^{xH} S^{-1}$ | (Ensemble) Kalman gain |
| $U = (HX)'^\top / \sqrt{N_e - 1}$ | Low-rank factor of the ensemble term in $S$ |

## Transforms and dynamics

| Symbol | Description |
|---|---|
| $\mathcal{M}$, $M$ | Forecast model (dynamics); $M_z$ a latent surrogate |
| $\tilde{C} = (N_e - 1) I + Y' R^{-1} Y'^\top$ | ETKF ensemble-space transform precision |
| $W$ (also $W_a$) | Ensemble transform / perturbation-weight matrix |
| $\bar{w}$ | Mean-update weight vector in ensemble space |
| $\Theta$ | Mean-preserving random rotation (ETKF-Livings) |
| $E$, $D$ | Encoder / decoder of a latent map; $z = E(x)$, $x \approx D(z)$ |
| $\theta$ | Trainable parameters (chapter 14) |
| $\mathcal{L}$, $L_t$ | Training loss; per-step local loss (ROAD-EnKF) |

## Localization and inflation

| Symbol | Description |
|---|---|
| $\rho$ | Localization taper (function or matrix); entries in $[0, 1]$ |
| $r$ | Localization half-width; compactly supported tapers vanish at $2r$ |
| $\circ$ | Schur (Hadamard, element-wise) matrix product |
| $\rho^{xy}$, $\rho^{yy}$ | State–obs and obs–obs taper matrices |
| $d_{ij}$ | Physical distance between variables $i$ and $j$ |
| $\lambda$ | Multiplicative inflation factor ($X' \leftarrow \lambda X'$) |
| $\alpha$ | Relaxation coefficient (RTPS / RTPP), in $[0, 1]$ |
| $Q_{\text{add}}$ | Additive-inflation (model-error) covariance |
| $\lambda^*$, $\mu$ | Ledoit-Wolf shrinkage intensity; average sample eigenvalue $\operatorname{tr}(P)/N_x$ |

## Diagnostics

| Symbol | Description |
|---|---|
| $\chi^2 = d^\top S^{-1} d$ | Innovation consistency statistic; $E[\chi^2] = N_y$ |
| $d_f$, $d_a$ | Forecast / analysis departures (Desroziers) |
| $\mathrm{SSR}$ | Spread-skill ratio (RMS spread / RMSE of the mean) |
| $\mathrm{DFS} = \operatorname{tr}(KH)$ | Degrees of freedom for signal |
| $\mathrm{CRPS}$ | Continuous Ranked Probability Score |
| $N_{\mathrm{eff}} = 1 / \sum_j w_j^2$ | Effective ensemble size |
| $A$ | Analysis-error covariance (in Desroziers identities) |

## General

| Symbol | Description |
|---|---|
| $\mathbf{1}$ | Vector of ones, $(N_e,)$ unless stated otherwise |
| $I$ | Identity matrix |
| $\mathcal{N}(\mu, \Sigma)$ | Gaussian distribution |
| $\varepsilon$ | Random perturbation / noise draw |
| $\lVert \cdot \rVert_F$ | Frobenius norm |
| $\operatorname{tr}(\cdot)$ | Matrix trace |
| $E[\cdot]$ | Expectation |
| $\mathbb{1}\{\cdot\}$ | Indicator function |
| $x^*$ | True state (twin / OSSE experiments) |

## filterax shape conventions

| Array | Shape |
|---|---|
| Ensemble (`particles`) | `(N_e, N_x)` |
| Observation-space ensemble | `(N_e, N_y)` |
| Observation vector | `(N_y,)` |
| Kalman gain | `(N_x, N_y)` |
| Taper matrices `rho_xy` / `rho_yy` | `(N_x, N_y)` / `(N_y, N_y)` |
| Stacked histories (`*_history`) | `(T, N_e, N_x)` |
| Stacked observations (differentiable loop) | `(T, N_y)` |
| Latent ensemble | `(N_e, N_z)` |
