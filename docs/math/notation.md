# Notation

Every chapter uses the symbols below: the letter says what kind of object a symbol is, the subscript says which variable it belongs to, and every symbol has a shape.
Equations use $\vect{x}$ in this documentation and Unicode, such as Σ_x|y, in code comments.
Docstrings have not moved to these symbols yet, and the last section lists what they still use.

The convention that matters most is that an ensemble stacks its members as rows, $\mat{X} \in \mathbb{R}^{N_\mathcal{E} \times D_x}$, so axis 0 of every ensemble array is the ensemble axis.
That matches the `(N_e, N_x)` shape of every filterax `particles` argument.

## Sizes and indices

| Symbol | Meaning | In code |
| --- | --- | --- |
| $D_x$, $D_y$, $D_z$, $D_\theta$ | length of the state, the observation vector, a latent state and a parameter vector | `N_x`, `N_y`, `N_z`, `N_p` |
| $N_\mathcal{E}$, $e$ | number of ensemble members, and one member $\vect{x}^{(e)}$ with $e \in \mathcal{E}$ | `N_e`, `J` |
| $T$, $t$ | number of assimilation windows, and one window | `T` |
| $N_\text{iter}$, $i$ | number of process iterations, and one iteration, written as a superscript $\gvect{\theta}^{(i)}$ | `n_iterations` |
| $\vect{s}$ | a spatial coordinate | `state_coords`, `obs_coords` |
| $\vect{1}$, $\mat{I}$ | the vector of ones and the identity, with sizes clear from context | |

## State and observations

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $\vect{x}_t$ | $\mathbb{R}^{D_x}$ | the state at time $t$ |
| $\vect{y}_t$ | $\mathbb{R}^{D_y}$ | the observations at time $t$ |
| $\vect{z}_t$ | $\mathbb{R}^{D_z}$ | a latent state |
| $\gvect{\theta}$ | $\mathbb{R}^{D_\theta}$ | parameters: the unknown of an inverse problem, or trainable weights |
| $\gvect{\varepsilon}_t$ | $\mathbb{R}^{D_y}$ | observation noise, $\gvect{\varepsilon}_t \sim \mathcal{N}(\vect{0}, \gmat{\Sigma}_\varepsilon)$ |
| $\gvect{\eta}_t$ | $\mathbb{R}^{D_x}$ | model noise, $\gvect{\eta}_t \sim \mathcal{N}(\vect{0}, \gmat{\Sigma}_\eta)$ |
| $\vect{x}_{t \mid t-1}$, $\vect{x}_{t \mid t}$, $\vect{x}_{t \mid T}$ | $\mathbb{R}^{D_x}$ | the forecast, the analysis and the smoothed state |

## Ensembles

A bar is a sample average, a tilde is an anomaly from it, and a hat is an estimate.

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $\mat{X}$ | $N_\mathcal{E} \times D_x$ | the state ensemble, with members $\vect{x}^{(e)}$ as rows |
| $\bar{\vect{x}} = \frac{1}{N_\mathcal{E}} \sum_e \vect{x}^{(e)}$ | $\mathbb{R}^{D_x}$ | the ensemble mean |
| $\tilde{\mat{X}} = \mat{X} - \vect{1} \bar{\vect{x}}^\top$ | $N_\mathcal{E} \times D_x$ | the state anomalies, with rows $\tilde{\vect{x}}^{(e)}$ |
| $\vect{h}^{(e)} = h(\vect{x}^{(e)})$ | $\mathbb{R}^{D_y}$ | one member mapped to observation space |
| $\bar{\vect{h}}$, $\tilde{\mat{H}}$ | $\mathbb{R}^{D_y}$, $N_\mathcal{E} \times D_y$ | their mean, and their anomalies as rows |
| $\bar{\vect{r}} = \vect{y} - \bar{\vect{h}}$ | $\mathbb{R}^{D_y}$ | the mean residual, also called the innovation |
| $\hat{\gmat{\Sigma}}_x = \tilde{\mat{X}}^\top \tilde{\mat{X}} / (N_\mathcal{E} - 1)$ | $D_x \times D_x$ | the sample covariance, never formed |
| $\hat{\gmat{\Sigma}}_{xy} = \tilde{\mat{X}}^\top \tilde{\mat{H}} / (N_\mathcal{E} - 1)$ | $D_x \times D_y$ | the sample cross-covariance |
| $\hat{\gmat{\Sigma}}_h = \tilde{\mat{H}}^\top \tilde{\mat{H}} / (N_\mathcal{E} - 1)$ | $D_y \times D_y$ | the sample covariance of the mapped members |

## Covariances and gains

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $\gmat{\Sigma}_x$, $\gvect{\mu}_x$ | $D_x \times D_x$, $\mathbb{R}^{D_x}$ | the true covariance and mean of the state |
| $\gmat{\Sigma}_\varepsilon$ | $D_y \times D_y$ | the observation-error covariance |
| $\gmat{\Sigma}_\eta$ | $D_x \times D_x$ | the model-error covariance |
| $\gmat{\Sigma}_y = \mat{H} \gmat{\Sigma}_x \mat{H}^\top + \gmat{\Sigma}_\varepsilon$ | $D_y \times D_y$ | the covariance of the observations; $\hat{\gmat{\Sigma}}_y = \hat{\gmat{\Sigma}}_h + \gmat{\Sigma}_\varepsilon$ in the ensemble |
| $\mat{G} = \gmat{\Sigma}_{xy} \gmat{\Sigma}_y^{-1}$ | $D_x \times D_y$ | the gain |
| $\gmat{\Sigma}_{x \mid y}$ | $D_x \times D_x$ | the posterior covariance |

## Maps

| Symbol | Maps | Meaning |
| --- | --- | --- |
| $f$, $\mat{F}$ | $\mathbb{R}^{D_x} \to \mathbb{R}^{D_x}$ | the forward model, and its linear form |
| $h$, $\mat{H}$ | $\mathbb{R}^{D_x} \to \mathbb{R}^{D_y}$ | the observation operator, and its linear form |
| $g_\text{enc}$, $g_\text{dec}$ | $\mathbb{R}^{D_x} \to \mathbb{R}^{D_z}$ and back | a latent encoder and decoder |
| $\mathcal{L}$ | scalar | a training loss, always the quantity minimised |

## Ensemble transforms

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $\gmat{\Lambda}_w$, $\gmat{\Sigma}_w = \gmat{\Lambda}_w^{-1}$ | $N_\mathcal{E} \times N_\mathcal{E}$ | the precision and covariance of the ensemble-space weights |
| $\bar{\vect{w}}$ | $\mathbb{R}^{N_\mathcal{E}}$ | the mean weights |
| $\mat{W}$ | $N_\mathcal{E} \times N_\mathcal{E}$ | the symmetric transform |
| $\mat{Q}$ | $N_\mathcal{E} \times N_\mathcal{E}$ | a mean-preserving random rotation |

## Localization and inflation

| Symbol | Shape | Meaning |
| --- | --- | --- |
| $\rho$, $\rho_{xy}$, $\rho_{yy}$ | scalar, $D_x \times D_y$, $D_y \times D_y$ | a taper in $[0, 1]$, and the taper matrices it fills |
| $\ell$ | scalar | a taper's half-width, the `radius` argument; a compactly supported taper vanishes at $2 \ell$ |
| $d_{ij}$ | scalar | the distance between variables $i$ and $j$ |
| $\circ$ | | the Schur, or element-wise, product |
| $\zeta$ | scalar | a multiplicative inflation factor |
| $\alpha$ | scalar | a relaxation coefficient in $[0, 1]$, for RTPS and RTPP |
| $\lambda$ | scalar | the Ledoit-Wolf shrinkage intensity |

## Diagnostics

| Symbol | Meaning |
| --- | --- |
| $\chi^2 = \bar{\vect{r}}^\top \hat{\gmat{\Sigma}}_y^{-1} \bar{\vect{r}}$ | the innovation consistency statistic, with $\mathbb{E}[\chi^2] = D_y$ |
| $\mathrm{SSR}$ | the spread-skill ratio, the RMS spread over the RMSE of the mean |
| $\mathrm{DFS} = \op{tr}(\mat{G} \mat{H})$ | the degrees of freedom for signal |
| $\mathrm{CRPS}$ | the continuous ranked probability score |
| $N_\text{eff} = 1 / \sum_e w_e^2$ | the effective ensemble size, for importance weights $w_e$ |
| $\vect{x}_\text{true}$ | the true state in a twin experiment |

## Classic letters

Papers on data assimilation use the letters on the left, and so do filterax's docstrings for now.

| Classic | Here | Meaning |
| --- | --- | --- |
| $N_e$, $J$ | $N_\mathcal{E}$ | the ensemble size |
| $N_x$, $N_y$, $N_z$ | $D_x$, $D_y$, $D_z$ | the sizes of the state, the observations and a latent state |
| $N_p$, $N_d$ | $D_\theta$, $D_y$ | the sizes of the parameters and the data in an inverse problem |
| $X'$, $Y'$ | $\tilde{\mat{X}}$, $\tilde{\mat{H}}$ | the anomalies |
| $P$, $P^f$, $B$ | $\hat{\gmat{\Sigma}}_x$, or $\gmat{\Sigma}_x$ when it is the true covariance | the forecast covariance |
| $C^{xH}$, $C^{HH}$ | $\hat{\gmat{\Sigma}}_{xy}$, $\hat{\gmat{\Sigma}}_h$ | the sample covariances |
| $R$, $\Gamma$ | $\gmat{\Sigma}_\varepsilon$ | the observation-error covariance |
| $Q$ | $\gmat{\Sigma}_\eta$ | the model-error covariance |
| $S$ | $\gmat{\Sigma}_y$ | the innovation covariance |
| $K$ | $\mat{G}$ | the Kalman gain |
| $d$, $v$ | $\bar{\vect{r}}$ | the innovation |
| $\vect{x}^f$, $\vect{x}^a$, $\vect{x}^s$ | $\vect{x}_{t \mid t-1}$, $\vect{x}_{t \mid t}$, $\vect{x}_{t \mid T}$ | the forecast, analysis and smoothed states |
| $M$, $\mathcal{M}$ | $f$, $\mat{F}$ | the dynamical model |
| $\mathcal{H}$ | $h$, $\mat{H}$ | the observation operator |
| $\tilde{C}$, $T$ | $\gmat{\Lambda}_w$, $\gmat{\Sigma}_w$ | the transform precision and its inverse |
| $\Theta$ | $\mat{Q}$ | the mean-preserving rotation |
| $E$, $D$ | $g_\text{enc}$, $g_\text{dec}$ | a latent encoder and decoder |
| $r$ | $\ell$ | the localization half-width |
| $\lambda$ | $\zeta$ | the multiplicative inflation factor |

## Array shapes

| Array | Shape |
| --- | --- |
| Ensemble (`particles`) | `(N_e, N_x)` |
| Mapped ensemble (`obs_particles`) | `(N_e, N_y)` |
| Observation vector | `(N_y,)` |
| Kalman gain | `(N_x, N_y)` |
| Taper matrices `rho_xy`, `rho_yy` | `(N_x, N_y)`, `(N_y, N_y)` |
| Stacked histories (`*_history`) | `(T, N_e, N_x)` |
| Stacked observations, in the differentiable loop | `(T, N_y)` |
| Latent ensemble | `(N_e, N_z)` |
