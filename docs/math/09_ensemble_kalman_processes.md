---
numbering:
  enumerator: "9.%s"
---

# Ensemble Kalman processes: EKI, ETKI, GNKI, TEKI, EKS and UKI

The filters of chapters 5 to 8 track a state through time.
The same algebra also solves a static inverse problem.
Given one batch of data, it estimates the parameters of a forward model that may be expensive and may have no derivatives.
An ensemble of candidate parameters is iterated, and its own covariances stand in for the Jacobian.

Each process in this chapter is one choice about the same tempered Kalman update, and that choice organises the chapter.
EKI moves every member with the same data, so the ensemble collapses onto a point estimate.
ETKI and UKI keep the Gaussian moments right, so a linear problem lands exactly on the posterior.
GNKI and TEKI bring the prior into the update and head for its mean or mode.
EKS adds noise, so the ensemble samples the posterior and never stops.
A scheduler then decides how far each iteration goes.

## The inverse problem

The data are one observation vector $\vect{y} \in \mathbb{R}^{D_y}$ from

$$\vect{y} = h(\gvect{\theta}) + \gvect{\varepsilon}, \qquad \gvect{\varepsilon} \sim \mathcal{N}(\vect{0}, \gmat{\Sigma}_\varepsilon), \qquad \gvect{\theta} \sim \mathcal{N}(\gvect{\mu}_\theta, \gmat{\Sigma}_\theta),$$ (eq-processes-problem)

with parameters $\gvect{\theta} \in \mathbb{R}^{D_\theta}$, a forward model $h : \mathbb{R}^{D_\theta} \to \mathbb{R}^{D_y}$, a noise covariance $\gmat{\Sigma}_\varepsilon \in \mathbb{R}^{D_y \times D_y}$, and a prior mean $\gvect{\mu}_\theta \in \mathbb{R}^{D_\theta}$ and covariance $\gmat{\Sigma}_\theta \in \mathbb{R}^{D_\theta \times D_\theta}$.
Docstrings call the forward model $G$ and the noise $\Gamma$, but $\mat{G}$ is the gain here.

Stack the members as rows of $\gmat{\Theta} \in \mathbb{R}^{N_\mathcal{E} \times D_\theta}$, with mean $\bar{\gvect{\theta}} \in \mathbb{R}^{D_\theta}$ and anomalies $\tilde{\gmat{\Theta}} \in \mathbb{R}^{N_\mathcal{E} \times D_\theta}$.
Each member maps to $\vect{h}^{(e)} = h(\gvect{\theta}^{(e)})$, with mean $\bar{\vect{h}} \in \mathbb{R}^{D_y}$ and anomalies $\tilde{\mat{H}} \in \mathbb{R}^{N_\mathcal{E} \times D_y}$.
With $c = N_\mathcal{E} - 1$, the sample covariances are $\hat{\gmat{\Sigma}}_\theta = \tilde{\gmat{\Theta}}^\top \tilde{\gmat{\Theta}} / c$, $\hat{\gmat{\Sigma}}_{\theta y} = \tilde{\gmat{\Theta}}^\top \tilde{\mat{H}} / c$ and $\hat{\gmat{\Sigma}}_h = \tilde{\mat{H}}^\top \tilde{\mat{H}} / c$.

An iteration $i$ takes a step $\Delta t^{(i)} > 0$ in an artificial time $\tau^{(i)} = \sum_{k < i} \Delta t^{(k)}$, which the code calls `algo_time`.
Tempering interpolates the prior at $\tau = 0$ and the posterior at $\tau = 1$:

$$p_\tau(\gvect{\theta} \mid \vect{y}) \propto p(\gvect{\theta}) p(\vect{y} \mid \gvect{\theta})^\tau.$$ (eq-processes-tempered)

Raising a Gaussian likelihood to the power $\Delta t$ replaces $\gmat{\Sigma}_\varepsilon$ with $\gmat{\Sigma}_\varepsilon / \Delta t$.

## EKI: one data vector for every member

Ensemble Kalman inversion {cite:p}`iglesias2013ensemble` applies a tempered Kalman update to the ensemble.
filterax passes the same $\vect{y}$ to every member, so the step has no randomness:

$$\hat{\mat{G}} = \hat{\gmat{\Sigma}}_{\theta y} \left(\hat{\gmat{\Sigma}}_h + \gmat{\Sigma}_\varepsilon / \Delta t\right)^{-1}, \qquad \gvect{\theta}^{(e)} \leftarrow \gvect{\theta}^{(e)} + \hat{\mat{G}} \left(\vect{y} - \vect{h}^{(e)}\right).$$ (eq-processes-eki)

Here $\hat{\mat{G}} \in \mathbb{R}^{D_\theta \times D_y}$, and $h$ is only ever evaluated, never differentiated.
The step is `gaussx.eki_step`, which forms the bracket densely when $N_\mathcal{E} \ge D_y$ and applies Woodbury otherwise, and $\gmat{\Sigma}_\varepsilon / \Delta t$ stays a lazily scaled operator.
As $\Delta t \to 0$ the increment tends to $\Delta t \hat{\gmat{\Sigma}}_{\theta y} \gmat{\Sigma}_\varepsilon^{-1} (\vect{y} - \vect{h}^{(e)})$, the Euler step of a flow in $\tau$.
For a linear $h(\gvect{\theta}) = \mat{H} \gvect{\theta}$ we have $\hat{\gmat{\Sigma}}_{\theta y} = \hat{\gmat{\Sigma}}_\theta \mat{H}^\top$, so the flow is a gradient descent on the data misfit, preconditioned by $\hat{\gmat{\Sigma}}_\theta$.

Every increment is a combination of the anomalies, so the members never leave the affine span of the initial ensemble.
That is why EKI runs with $N_\mathcal{E} \le D_\theta$, and also why it can recover nothing outside that span.
[](#code-processes-span) checks the property with 5 members and $D_\theta = 10$.

### EKI does not follow the tempered posterior

The common $\vect{y}$ has a cost.
Subtract the mean of [](#eq-processes-eki) from a linear $h$ and the anomalies obey $\tilde{\gmat{\Theta}} \leftarrow \tilde{\gmat{\Theta}} (\mat{I} - \hat{\mat{G}} \mat{H})^\top$, which gives

$$\hat{\gmat{\Sigma}}_\theta \leftarrow (\mat{I} - \hat{\mat{G}} \mat{H}) \hat{\gmat{\Sigma}}_\theta (\mat{I} - \hat{\mat{G}} \mat{H})^\top,$$ (eq-processes-eki-anomalies)

where the Kalman update would give $(\mat{I} - \hat{\mat{G}} \mat{H}) \hat{\gmat{\Sigma}}_\theta$.
The missing term is the one that perturbed observations would add.
So one step with $\Delta t = 1$ gets the Kalman mean right, and contracts the covariance too far.

Several small steps get both wrong.
Write the data precision as $\gmat{\Lambda}_h = \mat{H}^\top \gmat{\Sigma}_\varepsilon^{-1} \mat{H} \in \mathbb{R}^{D_\theta \times D_\theta}$.
Expanding [](#eq-processes-eki-anomalies) to first order in $\Delta t$ gives $\mathrm{d} \hat{\gmat{\Sigma}}_\theta / \mathrm{d} \tau = -2 \hat{\gmat{\Sigma}}_\theta \gmat{\Lambda}_h \hat{\gmat{\Sigma}}_\theta$, and that integrates to

$$\hat{\gmat{\Sigma}}_\theta(\tau)^{-1} = \hat{\gmat{\Sigma}}_\theta(0)^{-1} + 2 \tau \gmat{\Lambda}_h,$$ (eq-processes-eki-precision)

with the data counted twice over.
The mean follows in closed form when $\gmat{\Lambda}_h$ is invertible.
Let $\gvect{\theta}_\text{LS}$ be the least-squares fit, let $\hat{\gmat{\Sigma}}_\theta(0) = \mat{L} \mat{L}^\top$, and let $\mat{L}^\top \gmat{\Lambda}_h \mat{L}$ have eigenvalues $\gamma_d$ and eigenvectors $\vect{q}_d$.
The component $z_d = \vect{q}_d^\top \mat{L}^{-1} (\bar{\gvect{\theta}} - \gvect{\theta}_\text{LS})$ obeys $\mathrm{d} z_d / \mathrm{d} \tau = - \gamma_d z_d / (1 + 2 \tau \gamma_d)$, so at $\tau = 1$

$$z_d^\text{EKI} = \frac{z_d(0)}{\sqrt{1 + 2 \gamma_d}}, \qquad z_d^\text{Kalman} = \frac{z_d(0)}{1 + \gamma_d}.$$ (eq-processes-eki-limit)

So EKI with many small steps does not reach the Kalman answer: its covariance is too small, while its mean stops short because $\sqrt{1 + 2 \gamma_d} < 1 + \gamma_d$.
[](#code-processes-agreement) measures both.
EKI is a regularised point estimator {cite:p}`iglesias2016regularizing,schillings2017analysis`, and its final spread is not a posterior uncertainty.

## ETKI: the moments stay right

Ensemble transform Kalman inversion replaces the common $\vect{y}$ with the deterministic transform of [chapter 6](06_etkf.md), at the tempered noise.
The mean moves as in [](#eq-processes-eki), and the anomalies go through a symmetric root:

$$\bar{\gvect{\theta}} \leftarrow \bar{\gvect{\theta}} + \hat{\mat{G}} (\vect{y} - \bar{\vect{h}}), \qquad \tilde{\gmat{\Theta}} \leftarrow \sqrt{c} \left(c \mat{I} + \Delta t \tilde{\mat{H}} \gmat{\Sigma}_\varepsilon^{-1} \tilde{\mat{H}}^\top\right)^{-1/2} \tilde{\gmat{\Theta}}.$$ (eq-processes-etki)

By [](#eq-etkf-covariance) the sample covariance after the step is exactly the Kalman one.
For a linear $h$ and an invertible $\hat{\gmat{\Sigma}}_\theta$ that is a sum in precision form,

$$\hat{\gmat{\Sigma}}_\theta^{-1} \leftarrow \hat{\gmat{\Sigma}}_\theta^{-1} + \Delta t \gmat{\Lambda}_h, \qquad \hat{\gmat{\Sigma}}_\theta^{-1} \bar{\gvect{\theta}} \leftarrow \hat{\gmat{\Sigma}}_\theta^{-1} \bar{\gvect{\theta}} + \Delta t \mat{H}^\top \gmat{\Sigma}_\varepsilon^{-1} \vect{y},$$ (eq-processes-etki-precision)

so any schedule with $\sum_i \Delta t^{(i)} = 1$ composes to the single Kalman update of the initial mean and sample covariance.
This is the tempering of [](#eq-processes-tempered) done exactly, where [](#eq-processes-eki-precision) has a 2 in it.

## UKI: sigma points instead of an ensemble

Unscented Kalman inversion {cite:p}`huang2022iterated` keeps a Gaussian belief $\mathcal{N}(\gvect{\mu}_\theta, \gmat{\Sigma}_\theta)$, which starts at the prior and is overwritten at each step.
It probes $h$ at $2 D_\theta + 1$ deterministic sigma points {cite:p}`wan2000unscented`:

$$\gvect{\chi}_0 = \gvect{\mu}_\theta, \qquad \gvect{\chi}_{\pm d} = \gvect{\mu}_\theta \pm \sqrt{D_\theta + \lambda} [\gmat{\Sigma}_\theta^{1/2}]_d, \qquad \lambda = \alpha^2 (D_\theta + \kappa) - D_\theta,$$ (eq-processes-sigma)

where $[\cdot]_d$ is column $d$ of the principal root.
The mean weights are $w_{m,0} = \lambda / (D_\theta + \lambda)$ and $w_{m,k} = 1 / (2 D_\theta + 2 \lambda)$, and the covariance weights differ only at the centre, with $w_{c,0} = w_{m,0} + 1 - \alpha^2 + \beta$.
With $\vect{h}_k = h(\gvect{\chi}_k)$, the weighted moments are $\bar{\vect{h}} = \sum_k w_{m,k} \vect{h}_k$, $\gmat{\Sigma}_h = \sum_k w_{c,k} (\vect{h}_k - \bar{\vect{h}}) (\vect{h}_k - \bar{\vect{h}})^\top$ and $\gmat{\Sigma}_{\theta y} = \sum_k w_{c,k} (\gvect{\chi}_k - \gvect{\mu}_\theta) (\vect{h}_k - \bar{\vect{h}})^\top$, and the update is

$$\gmat{\Sigma}_y = \gmat{\Sigma}_h + \gmat{\Sigma}_\varepsilon / \Delta t, \qquad \gvect{\mu}_\theta \leftarrow \gvect{\mu}_\theta + \gmat{\Sigma}_{\theta y} \gmat{\Sigma}_y^{-1} (\vect{y} - \bar{\vect{h}}), \qquad \gmat{\Sigma}_\theta \leftarrow \gmat{\Sigma}_\theta - \gmat{\Sigma}_{\theta y} \gmat{\Sigma}_y^{-1} \gmat{\Sigma}_{\theta y}^\top.$$ (eq-processes-uki)

filterax calls `gaussx.sigma_point_kalman_update(..., joseph=False)`, which is this standard form.
The Joseph form that gaussx offers rests on the statistical linearisation $\mat{H} = \gmat{\Sigma}_{\theta y}^\top \gmat{\Sigma}_\theta^{-1}$, and expanding it with $\gmat{\Sigma}_{\theta y} = \mat{G} \gmat{\Sigma}_y$ gives the same matrix, so the flag only changes the rounding.
The sigma-point moments are exact for a linear $h$, so [](#eq-processes-etki-precision) holds and the schedule lands on the posterior.
The parametric methods `init_parametric` and `update_parametric` hold the belief directly, while `init` and `update` carry it as sigma points in the `particles` field, to fit the common protocol.

Two things differ from what the name may suggest.
filterax has the default $\alpha = 1$, where $w_{m,0} = 0$, rather than the small $\alpha$ common in the unscented literature, for a reason measured in [](#code-processes-alpha).
And it has no prediction step: the artificial dynamics of {cite:t}`huang2022iterated`, which keep their covariance from collapsing, are not implemented, and only tempering remains.
So $\tau = 1$ is where the belief is right, and the update keeps contracting past it, as [](#code-processes-uki-past) shows.
The Layer 2 `UKI` stops at $\tau \ge 1$, and a Layer 1 loop has to do the same.

## GNKI: a Gauss-Newton step to the posterior mean

Gauss-Newton Kalman inversion estimates the Jacobian from the ensemble and takes a Gauss-Newton step with a pull towards the prior.
It solves against $\hat{\gmat{\Sigma}}_\theta$, so it needs $N_\mathcal{E} > D_\theta$ and raises an error otherwise:

$$\hat{\mat{J}}_h = \hat{\gmat{\Sigma}}_{\theta y}^\top \left(\hat{\gmat{\Sigma}}_\theta + \delta \mat{I}\right)^{-1}, \qquad \mat{G} = \left(\hat{\mat{J}}_h^\top \gmat{\Sigma}_\varepsilon^{-1} \hat{\mat{J}}_h + \gmat{\Sigma}_\theta^{-1}\right)^{-1} \hat{\mat{J}}_h^\top \gmat{\Sigma}_\varepsilon^{-1},$$ (eq-processes-gnki-gain)

$$\gvect{\theta}^{(e)} \leftarrow \gvect{\theta}^{(e)} + \Delta t \left[\mat{G} (\vect{y} - \vect{h}^{(e)}) + (\mat{I} - \mat{G} \hat{\mat{J}}_h) (\gvect{\mu}_\theta - \gvect{\theta}^{(e)})\right],$$ (eq-processes-gnki)

with $\hat{\mat{J}}_h \in \mathbb{R}^{D_y \times D_\theta}$ and an absolute jitter $\delta$, by default $10^{-6}$.
Take a linear $h$ with $\delta = 0$, so that $\hat{\mat{J}}_h = \mat{H}$, and collect the terms in $\gvect{\theta}^{(e)}$.
They cancel, and the bracket is $\gvect{\mu}_{\theta \mid y} - \gvect{\theta}^{(e)}$, where $\gvect{\mu}_{\theta \mid y} = \gvect{\mu}_\theta + \mat{G} (\vect{y} - \mat{H} \gvect{\mu}_\theta)$ is the exact posterior mean.
Every member therefore contracts onto that one point at the same rate:

$$\gvect{\theta}^{(e)} - \gvect{\mu}_{\theta \mid y} \leftarrow (1 - \Delta t) \left(\gvect{\theta}^{(e)} - \gvect{\mu}_{\theta \mid y}\right).$$ (eq-processes-gnki-contraction)

GNKI recovers the posterior mean, in one step at $\Delta t = 1$, and its spread is not a posterior covariance.
The GNKI docstring claims the exact covariance as well, and [](#code-processes-gnki) shows otherwise.

## TEKI: the prior as data

Tikhonov EKI {cite:p}`chada2020tikhonov` runs [](#eq-processes-eki) on a system that observes the parameters too:

$$\tilde{h}(\gvect{\theta}) = \begin{bmatrix} h(\gvect{\theta}) \\ \gvect{\theta} \end{bmatrix}, \qquad \tilde{\vect{y}} = \begin{bmatrix} \vect{y} \\ \gvect{\mu}_\theta \end{bmatrix}, \qquad \tilde{\gmat{\Sigma}}_\varepsilon = \begin{bmatrix} \gmat{\Sigma}_\varepsilon & \mat{0} \\ \mat{0} & \gmat{\Sigma}_\theta \end{bmatrix},$$ (eq-processes-teki)

through `gaussx.tikhonov_augment`, so $\gmat{\Sigma}_\theta$ stays an operator inside a block-diagonal one.
The least-squares cost of the augmented system is the negative log-posterior,

$$\mathcal{J}(\gvect{\theta}) = \frac{1}{2} \Vert \vect{y} - h(\gvect{\theta}) \Vert^2_{\gmat{\Sigma}_\varepsilon^{-1}} + \frac{1}{2} \Vert \gvect{\theta} - \gvect{\mu}_\theta \Vert^2_{\gmat{\Sigma}_\theta^{-1}},$$ (eq-processes-teki-cost)

and for a linear $h$ its minimiser $\gvect{\theta}^\star$ is a fixed point of the mean, since the normal equations make the projected residual vanish there.
So the prior is not forgotten, and the run has no reason to stop at $\tau = 1$.

The convergence is slow, and [](#eq-processes-eki-limit) says why.
The update is still EKI, and once the spread is small a step of $\Delta t = 1$ is a small step of the flow.
With $\gamma_d$ now from the augmented data precision, a component of the error falls as $(1 + 2 i \gamma_d)^{-1/2}$.
The mean reaches the mode at rate $i^{-1/2}$, and the spread falls at the same rate.
[](#code-processes-teki) measures it.

## SparseInversion: EKI with a threshold

`SparseInversion` follows the EKI step with a soft threshold, after {cite:t}`schneider2022ensemble`:

$$\gvect{\theta}^{(e)}_d \leftarrow \op{sign}(\gvect{\theta}^{(e)}_d) \max\left(\vert \gvect{\theta}^{(e)}_d \vert - \varrho \Delta t, 0\right),$$ (eq-processes-sparse)

with a penalty $\varrho > 0$ from the `penalty_weight` argument.
This sets inactive parameters exactly to zero.
It also takes $\varrho \Delta t$ from every active parameter at every step, while the EKI step shrinks with the spread, so the run length matters, as [](#code-processes-sparse) measures.

## EKS: sampling instead of collapse

The ensemble Kalman sampler in its ALDI form {cite:p}`garbunoinigo2020affine` adds noise and a correction to the continuous flow.
That makes it an interacting Langevin diffusion, with the posterior as its stationary law:

$$\mathrm{d} \gvect{\theta}^{(e)} = \left[\hat{\gmat{\Sigma}}_{\theta y} \gmat{\Sigma}_\varepsilon^{-1} (\vect{y} - \vect{h}^{(e)}) - \hat{\gmat{\Sigma}}_\theta \gmat{\Sigma}_\theta^{-1} (\gvect{\theta}^{(e)} - \gvect{\mu}_\theta) - \frac{D_\theta + 1}{N_\mathcal{E}} (\gvect{\theta}^{(e)} - \bar{\gvect{\theta}})\right] \mathrm{d} \tau + \sqrt{2 \hat{\gmat{\Sigma}}_\theta} \mathrm{d} \vect{w}^{(e)},$$ (eq-processes-eks)

where $\vect{w}^{(e)} \in \mathbb{R}^{D_\theta}$ is a Brownian motion.
The first term is the EKI drift, the second is the prior, the third corrects for the finite ensemble, and the noise keeps the spread alive.
With no prior arguments the second term is absent and the stationary law is the likelihood alone.

The discrete step has four parts:

1. The EKI step [](#eq-processes-eki), whose tempered gain acts as an implicit Euler step on the data term.
2. The explicit pull $-\Delta t (D_\theta + 1) / N_\mathcal{E} \tilde{\gmat{\Theta}}$.
3. The noise, drawn in ensemble space. With $\mat{N} \in \mathbb{R}^{N_\mathcal{E} \times N_\mathcal{E}}$ of standard normals, adding $\sqrt{2 \Delta t / c} \mat{N} \tilde{\gmat{\Theta}}$ gives each member the covariance $2 \Delta t \hat{\gmat{\Sigma}}_\theta$, and no $D_\theta \times D_\theta$ Cholesky factor is formed.
4. The prior, taken implicitly as $\gvect{\theta} \leftarrow \gvect{\theta} - (\mat{I} + \Delta t \hat{\gmat{\Sigma}}_\theta \gmat{\Sigma}_\theta^{-1})^{-1} \Delta t \hat{\gmat{\Sigma}}_\theta \gmat{\Sigma}_\theta^{-1} (\gvect{\theta} - \gvect{\mu}_\theta)$. A push-through identity reduces the solve to $N_\mathcal{E} \times N_\mathcal{E}$, and a prior much tighter than the ensemble then cannot destabilise the step.

### The step sets the bias

Take a scalar linear model with a flat prior, large $N_\mathcal{E}$ and precision $\gamma = \mat{H}^\top \gmat{\Sigma}_\varepsilon^{-1} \mat{H}$.
Over one step the data term divides an anomaly by $1 + \Delta t \hat{\Sigma}_\theta \gamma$, and the noise adds $2 \Delta t \hat{\Sigma}_\theta$ to the variance, so a stationary variance solves

$$\hat{\Sigma}_\theta = \frac{\hat{\Sigma}_\theta}{(1 + \Delta t \hat{\Sigma}_\theta \gamma)^2} + 2 \Delta t \hat{\Sigma}_\theta.$$ (eq-processes-eks-stationary)

Dividing by $\hat{\Sigma}_\theta$ and solving gives

$$\hat{\Sigma}_\theta \gamma = \frac{(1 - 2 \Delta t)^{-1/2} - 1}{\Delta t} \approx 1 + \frac{3}{2} \Delta t,$$ (eq-processes-eks-bias)

and no stationary solution for $\Delta t \ge 1 / 2$, where the variance grows at every step.
[](#code-processes-eks-step) confirms both.
So EKS samples a biased posterior, by an amount the step controls, and that is the job of its scheduler.

## Schedulers

Every process takes its $\Delta t^{(i)}$ from an `AbstractScheduler`, which reads the current `ProcessState` and returns a scalar.

`FixedScheduler(dt)` returns a constant, which you tune.
With `dt=1.0` one step of EKI or UKI is the full Kalman update, and the Layer 2 run stops there.

`DataMisfitController` is the rule of {cite:t}`iglesias2021adaptive`, with no tuning parameter.
With the cost of each member, $\mathcal{J}^{(e)} = \frac{1}{2} \Vert \vect{y} - \vect{h}^{(e)} \Vert^2_{\gmat{\Sigma}_\varepsilon^{-1}}$, and its ensemble mean $\bar{\mathcal{J}}$ and variance $\sigma_\mathcal{J}^2$,

$$\Delta t^{(i)} = \min\left(\max\left(\frac{D_y}{2 \bar{\mathcal{J}}}, \sqrt{\frac{D_y}{2 \sigma_\mathcal{J}^2}}\right), 1 - \tau^{(i)}\right).$$ (eq-processes-dmc)

Both candidates come from $2 \mathcal{J}$ being $\chi^2$ with $D_y$ degrees of freedom under the correct model, which has mean $D_y$ and variance $2 D_y$.
The steps are small while the members are far from the data, they grow as the misfit falls, and the clamp lands $\tau$ exactly on 1.
After that the rule returns $\Delta t = 0$, and filterax floors the step at $10^{-30}$, so that further calls change nothing and give no `NaN`.
The computation is `gaussx.discrepancy_step_size`, which applies $\gmat{\Sigma}_\varepsilon^{-1}$ through a structured solve.

`EKSStableScheduler` bounds the step by the ensemble covariance and by the stiffness of the posterior:

$$\Delta t^{(i)} = \min\left(\Delta t_\text{max}, \frac{a}{\Vert \hat{\gmat{\Sigma}}_\theta \Vert_F}, \frac{b}{\Vert \mat{S} \Vert_F}\right), \qquad \mat{S} = \frac{1}{c} \left(\tilde{\mat{H}} \gmat{\Sigma}_\varepsilon^{-1} \tilde{\mat{H}}^\top + \tilde{\gmat{\Theta}} \gmat{\Sigma}_\theta^{-1} \tilde{\gmat{\Theta}}^\top\right),$$ (eq-processes-eks-stable)

with $\mat{S} \in \mathbb{R}^{N_\mathcal{E} \times N_\mathcal{E}}$, where $\Delta t_\text{max}$, $a$ and $b$ are the arguments `max_dt`, `target` and `target_stiffness`.
The prior term in $\mat{S}$ appears only when the state carries a prior.
For a linear $h$, $\mat{S}$ has the nonzero spectrum of $\hat{\gmat{\Sigma}}_\theta (\gmat{\Lambda}_h + \gmat{\Sigma}_\theta^{-1})$, so at the exact posterior $\Vert \mat{S} \Vert_F = \sqrt{D_\theta}$ whatever the scale of the problem.
The step then settles at about $b / \sqrt{D_\theta}$, and the default $b = 0.1$ aims through [](#eq-processes-eks-bias) for an excess variance of about 10% on a low-dimensional, data-dominated problem.

## Implementation in filterax

The processes come at three levels, built on one set of update rules.

| Layer | Offers | Who runs $h$ |
| --- | --- | --- |
| `filterax.processes` | `EKI`, `ETKI`, `GNKI`, `TEKI`, `SparseInversion`, `EKS_Process` and `UKI`, each with `init` and `update` | you, so you control batching, checkpoints and dispatch |
| `filterax` | `EKI`, `TEKI`, `EKS` and `UKI`, each with `run` | the run loop, through `vmap` |
| `filterax.optax` | `eki`, `eks` and `uki`, as `optax.GradientTransformation` | the transformation, ignoring `grads` |

The Layer 2 `EKI` and `UKI` default to `DataMisfitController`, stop at $\tau \ge 1$, and allow 50 iterations.
`TEKI` and `EKS` always run their full budget, 200 and 500 iterations by default, because neither has a reason to stop at $\tau = 1$.
Passing a [`ProcessConfig`](https://jejjohnson.github.io/filterax/reference/processes/) overrides the scheduler and the budget together.
ETKI, GNKI and SparseInversion have no Layer 2 wrapper.

The example is the linear problem used through this chapter, with $D_\theta = 2$, $D_y = 3$ and 200 members drawn from a wide prior.

```{code-block} python
:caption: Layer 2 EKI and UKI, each stopped by the adaptive scheduler at $\tau = 1$.
:label: code-processes-layer2
import einx
import filterax as flx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
from jaxtyping import Array, Float

jax.config.update("jax_enable_x64", True)

# Shapes: N_E = 200 members; D_θ = 2 parameters; D_y = 3 observations

# H
forward_map = jnp.array([[1.0, 0.5], [-0.3, 1.2], [0.7, 0.1]])  # → (D_y, D_θ)
# y = H θ_true + ε
observation = forward_map @ jnp.array([1.0, -1.0]) + jnp.array([0.2, -0.3, 0.1])
# today (lineax)
# Σ_ε = 0.1 I
noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(3))  # (D_y, D_y) operator
# θ⁽ᵉ⁾ ~ 𝒩(𝟎, 4 I)
members = 2.0 * jr.normal(jr.key(0), (200, 2))  # → (N_E, D_θ)


# user
# h(θ) = H θ
def forward(theta: Float[Array, " param"]) -> Float[Array, " obs"]:
    return forward_map @ theta


# today (filterax)
eki_model = flx.EKI(forward_fn=forward, obs=observation, noise_cov=noise)
eki_result = eki_model.run(members)
print(eki_result.n_iterations, eki_result.converged)
print(eki_result.history_algo_time.round(4))
print(eki_result.mean.round(4))

# x̄ and Σ̂ of the initial ensemble, as the belief UKI starts from
start_mean = einx.mean("[member] param", members)  # → (D_θ,)
# today (filterax)
start_cov = flx.ensemble_covariance(members).as_matrix()  # → (D_θ, D_θ)
# today (lineax)
start_op = lx.MatrixLinearOperator(start_cov, tags=lx.positive_semidefinite_tag)
# today (filterax)
uki_model = flx.UKI(forward_fn=forward, obs=observation, noise_cov=noise)
uki_result = uki_model.run(start_mean, start_op)
print(uki_result.n_iterations, uki_result.mean.round(4))
```

It prints `4 True`, `[0.0169 0.0964 0.4656 1.    ]` and `[ 1.1542 -1.1028]` for EKI, then `4 [ 1.2307 -1.164 ]` for UKI.
The controller takes four steps, each several times the last, and lands on $\tau = 1$.
The two means differ because of [](#eq-processes-eki-limit), and only UKI's is the Kalman mean.

Layer 1 leaves the loop to you, and each `update` takes the forward evaluations.
It compiles, where the Layer 2 `EKI.run` calls `float` on $\tau$ and so raises a `ConcretizationTypeError` under `jax.jit`.
[](#code-processes-layer1) runs a fixed number of steps inside one `jax.lax.scan`, and later blocks reuse the helper.
The optax form keeps the ensemble in its state and returns the change in the mean as the update.
It ignores the gradient argument, so it composes with other optax transformations.

```{code-block} python
:caption: Layer 1 under a scan, and the same process as an optax transformation.
:label: code-processes-layer1
import optax


# user
def iterate(process, start, target, noise_cov, model, steps):
    # one init, then `steps` updates that each run the forward model on the members
    def body(state, _):
        # today (filterax)
        evaluations = jax.vmap(model)(state.particles)  # → (N_E, D_y)
        return process.update(state, evaluations), None

    # today (filterax)
    initial = process.init(start, target, noise_cov)
    final, _ = jax.lax.scan(body, initial, None, length=steps)
    return final


# today (filterax)
adaptive = flx.processes.EKI(scheduler=flx.DataMisfitController())
state = iterate(adaptive, members, observation, noise, forward, 4)
# θ̄
layer1_mean = einx.mean("[member] param", state.particles)  # → (D_θ,)
print(state.algo_time, jnp.abs(layer1_mean - eki_result.mean).max() < 1e-12)

# Further updates at τ = 1 take Δt = 0
evaluations = jax.vmap(forward)(state.particles)  # → (N_E, D_y)
# today (filterax)
after = adaptive.update(state, evaluations)
print(jnp.abs(after.particles - state.particles).max())

# today (filterax)
transform = flx.optax.eki(
    forward_fn=forward,
    obs=observation,
    noise_cov=noise,
    n_ensemble=200,
    init_spread=2.0,
)
params = jnp.zeros(2)  # → (D_θ,)
opt_state = transform.init(params)
for _ in range(4):
    updates, opt_state = transform.update(None, opt_state, params)
    # today (optax)
    params = optax.apply_updates(params, updates)
print(opt_state.algo_time, params.round(4))
```

It prints `1.0 True`, then `0.0`, then `1.0 [ 1.1523 -1.0971]`.
Layer 1 reproduces Layer 2, and another update at $\tau = 1$ moves nothing.
The optax result differs a little because `transform.init` draws its own ensemble around `params`.

## Algorithm

[](#code-processes-algorithm) writes out one step of [](#eq-processes-eki) in dense form, and compares it with the Layer 1 update at $\Delta t = 0.25$.

```{code-block} python
:caption: The EKI step [](#eq-processes-eki) written out in full, for a dense $\gmat{\Sigma}_\varepsilon$.
:label: code-processes-algorithm
# user
def eki_by_hand(
    particles: Float[Array, "member param"],
    mapped: Float[Array, "member obs"],
    target: Float[Array, " obs"],
    noise_covariance: Float[Array, "obs obs"],
    step: float,
) -> Float[Array, "member param"]:
    # Shapes: N_E members; D_θ parameters; D_y observations
    # c = N_E − 1
    prior_term = particles.shape[0] - 1.0
    # Θ̃ = Θ − 𝟏 θ̄ᵀ
    param_mean = einx.mean("[member] param", particles)  # → (D_θ,)
    param_anomalies = particles - param_mean  # → (N_E, D_θ)
    # H̃ = H − 𝟏 h̄ᵀ
    mapped_mean = einx.mean("[member] obs", mapped)  # → (D_y,)
    mapped_anomalies = mapped - mapped_mean  # → (N_E, D_y)
    # Σ̂_θy = Θ̃ᵀ H̃ / c
    cross = (
        einx.dot(  # → (D_θ, D_y)
            "member param, member obs -> param obs", param_anomalies, mapped_anomalies
        )
        / prior_term
    )
    # Σ̂_h = H̃ᵀ H̃ / c
    mapped_cov = (
        einx.dot(  # → (D_y, D_y)
            "member obs, member other -> obs other", mapped_anomalies, mapped_anomalies
        )
        / prior_term
    )
    # Σ̂_h + Σ_ε / Δt
    innovation = mapped_cov + noise_covariance / step  # → (D_y, D_y)
    # r⁽ᵉ⁾ = y − h⁽ᵉ⁾
    residuals = target - mapped  # → (N_E, D_y)
    # (Σ̂_h + Σ_ε / Δt)⁻¹ r⁽ᵉ⁾, one column per member
    whitened = jnp.linalg.solve(  # → (D_y, N_E)
        innovation, einx.id("member obs -> obs member", residuals)
    )
    # θ⁽ᵉ⁾ + Σ̂_θy (Σ̂_h + Σ_ε / Δt)⁻¹ r⁽ᵉ⁾
    return particles + einx.dot(  # → (N_E, D_θ)
        "param obs, obs member -> member param", cross, whitened
    )


mapped = jax.vmap(forward)(members)  # → (N_E, D_y)
by_hand = eki_by_hand(members, mapped, observation, 0.1 * jnp.eye(3), 0.25)
# today (filterax)
quarter = flx.processes.EKI(scheduler=flx.FixedScheduler(dt=0.25))
stepped = iterate(quarter, members, observation, noise, forward, 1)
print(jnp.abs(by_hand - stepped.particles).max() < 1e-12)
```

It prints `True`.

## Agreement with the Kalman update

With a linear $h$ the reference is a dense Kalman update of the initial mean and sample covariance, which [](#code-processes-agreement) computes in numpy along with the limit [](#eq-processes-eki-limit).
The table splits the budget $\sum_i \Delta t^{(i)} = 1$ into $N_\text{iter}$ equal steps.

```{code-block} python
:caption: EKI, ETKI and UKI against the dense Kalman update, and EKI against its closed-form limit.
:label: code-processes-agreement
# Shapes: N_E = 200; D_θ = 2; D_y = 3


# user
# θ̄ + Σ̂ Hᵀ (H Σ̂ Hᵀ + Σ_ε)⁻¹ (y − H θ̄), and the matching covariance
def dense_kalman(mean, covariance, matrix, target, variance):
    innovation = matrix @ covariance @ matrix.T + variance * np.eye(len(target))
    gain = np.linalg.solve(innovation, matrix @ covariance).T
    updated = mean + gain @ (target - matrix @ mean)
    return updated, covariance - gain @ matrix @ covariance


# user
# the limit of many small EKI steps, from z⁽ᵈ⁾ / √(1 + 2 γ_d) and Σ̂⁻¹ + 2 Λ_h
def eki_limit(mean, covariance, matrix, target, variance):
    precision = matrix.T @ matrix / variance
    fit = np.linalg.solve(precision, matrix.T @ target / variance)
    factor = np.linalg.cholesky(covariance)
    gammas, modes = np.linalg.eigh(factor.T @ precision @ factor)
    start = modes.T @ np.linalg.solve(factor, mean - fit)
    limit_mean = fit + factor @ modes @ (start / np.sqrt(1.0 + 2.0 * gammas))
    limit_scale = modes * (1.0 / (1.0 + 2.0 * gammas)) @ modes.T
    return limit_mean, factor @ limit_scale @ factor.T


reference_args = (
    np.asarray(start_mean),
    np.asarray(start_cov),
    np.asarray(forward_map),
    np.asarray(observation),
    0.1,
)
kalman_mean, kalman_cov = dense_kalman(*reference_args)
limit_mean, limit_cov = eki_limit(*reference_args)


# user
# the largest error in the mean, and the variance as a fraction of Kalman's
def score(ensemble):
    mean = einx.mean("[member] param", ensemble)  # → (D_θ,)
    # today (filterax)
    covariance = flx.ensemble_covariance(ensemble).as_matrix()  # → (D_θ, D_θ)
    error = jnp.abs(mean - kalman_mean).max()
    return error, jnp.trace(covariance) / np.trace(kalman_cov)


for steps in (1, 4, 16, 256):
    # today (filterax)
    schedule = flx.FixedScheduler(dt=1.0 / steps)
    eki_process = flx.processes.EKI(schedule)
    eki_end = iterate(eki_process, members, observation, noise, forward, steps)
    # today (filterax)
    etki_process = flx.processes.ETKI(schedule)
    etki_end = iterate(etki_process, members, observation, noise, forward, steps)
    # today (filterax)
    budget = flx.ProcessConfig(scheduler=schedule, n_iterations=steps)
    # today (filterax)
    uki_end = flx.UKI(
        forward_fn=forward, obs=observation, noise_cov=noise, config=budget
    ).run(start_mean, start_op)
    eki_error, eki_variance = score(eki_end.particles)
    etki_error, etki_variance = score(etki_end.particles)
    uki_error = jnp.abs(uki_end.mean - kalman_mean).max()
    uki_covariance = uki_end.covariance.as_matrix()  # → (D_θ, D_θ)
    uki_variance = jnp.trace(uki_covariance) / np.trace(kalman_cov)
    print(
        f"{steps:4d} EKI {eki_error:.3f} {eki_variance:.3f}"
        f" ETKI {etki_error:.0e} {etki_variance:.3f}"
        f" UKI {uki_error:.0e} {uki_variance:.3f}"
    )
    if steps == 256:
        eki_mean = einx.mean("[member] param", eki_end.particles)  # → (D_θ,)
        # today (filterax)
        eki_cov = flx.ensemble_covariance(eki_end.particles).as_matrix()
        mean_gap = jnp.abs(eki_mean - limit_mean).max()
        cov_gap = jnp.abs(eki_cov - limit_cov).max()
        print(f"limit {mean_gap:.1e} {cov_gap:.1e}")
```

It prints:

| $N_\text{iter}$ | EKI mean error | EKI variance | ETKI mean error | ETKI variance | UKI mean error | UKI variance |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `0.000` | `0.015` | `9e-16` | `1.000` | `2e-16` | `1.000` |
| 4 | `0.047` | `0.161` | `4e-16` | `1.000` | `2e-16` | `1.000` |
| 16 | `0.085` | `0.434` | `2e-16` | `1.000` | `4e-16` | `1.000` |
| 256 | `0.092` | `0.501` | `1e-15` | `1.000` | `1e-15` | `1.000` |

The variance columns are $\op{tr}(\hat{\gmat{\Sigma}}_\theta)$ over the Kalman value.
ETKI and UKI are exact for every split, as [](#eq-processes-etki-precision) says.
EKI has the Kalman mean only at one step, and its variance there is 1.5% of the Kalman variance.
With many small steps the variance tends to about a half, the value of [](#eq-processes-eki-precision) when the data dominate, and the mean error grows instead.
The last line prints `limit 2.7e-04 1.5e-04`, the gap between 256 EKI steps and the closed-form limit, in the mean and in the covariance.
UKI is given a `ProcessConfig`, because a schedule that needs more steps than the budget of 50 stops early with `converged` false.

[](#code-processes-span) checks the span property, and [](#code-processes-uki-past) takes UKI past $\tau = 1$.

```{code-block} python
:caption: EKI with fewer members than parameters stays in the initial span, and UKI past $\tau = 1$ keeps contracting.
:label: code-processes-span
# Shapes: N_E = 5 members; D_θ = 10 parameters; D_y = 4 observations

few = jr.normal(jr.key(7), (5, 10))  # → (N_E, D_θ)
wide_map = jr.normal(jr.key(8), (4, 10))  # → (D_y, D_θ)
wide_target = jnp.arange(4.0)  # → (D_y,)
# today (lineax)
wide_noise = lx.DiagonalLinearOperator(0.1 * jnp.ones(4))  # (D_y, D_y) operator


# user
def wide_forward(theta: Float[Array, " param"]) -> Float[Array, " obs"]:
    return wide_map @ theta


# today (filterax)
end = iterate(quarter, few, wide_target, wide_noise, wide_forward, 4)
few_mean = einx.mean("[member] param", few)  # → (D_θ,)
# Θ⁽ⁱ⁾ − 𝟏 θ̄⁽⁰⁾ᵀ
moved = end.particles - few_mean  # → (N_E, D_θ)
# an orthonormal basis of the rows of Θ̃⁽⁰⁾
_, _, basis = jnp.linalg.svd(few - few_mean, full_matrices=False)
basis = basis[:4]  # → (4, D_θ)
# the part of each member outside the span
inside = einx.dot("member param, mode param -> member mode", moved, basis)
outside = moved - einx.dot("member mode, mode param -> member param", inside, basis)
print(f"{jnp.abs(outside).max():.1e}")
```

It prints `1.3e-15`, so no member has left the span of the initial anomalies, though $D_\theta = 10$ and only 4 directions are available.

```{code-block} python
:caption: UKI on the linear problem, run past the point where the belief is right.
:label: code-processes-uki-past
# Shapes: D_θ = 2; D_y = 3; 2 D_θ + 1 = 5 sigma points

# today (filterax)
full = flx.processes.UKI(scheduler=flx.FixedScheduler(dt=1.0))
# today (filterax)
belief = full.init_parametric(start_mean, start_op, observation, noise)
for step in range(3):
    # today (filterax)
    points, _, _ = flx.processes.sigma_points(belief.mean, belief.covariance)
    # today (filterax)
    belief = full.update_parametric(
        belief,
        jax.vmap(forward)(points),
        obs=observation,
        noise_cov=noise,
        dt=jnp.asarray(1.0),
    )
    covariance = belief.covariance.as_matrix()  # → (D_θ, D_θ)
    print(step + 1, belief.mean.round(4), f"{jnp.trace(covariance):.4f}")
```

It prints `1 [ 1.2307 -1.164 ] 0.1223`, `2 [ 1.241 -1.171] 0.0616` and `3 [ 1.2445 -1.1734] 0.0412`, where the last number is $\op{tr}(\gmat{\Sigma}_\theta)$.
The first line is the Kalman answer, and the next two count the data again.

GNKI and TEKI have the prior in the update, so their reference is the exact posterior of [](#eq-processes-problem), not a Kalman update of the sample moments.
The prior here has $\gvect{\mu}_\theta = \vect{0}$ and $\gmat{\Sigma}_\theta = 4 \mat{I}$.

```{code-block} python
:caption: GNKI lands on the posterior mean with no spread, by the contraction [](#eq-processes-gnki-contraction).
:label: code-processes-gnki
# Shapes: N_E = 200; D_θ = 2; D_y = 3

prior_mean = jnp.zeros(2)  # → (D_θ,)
# today (lineax)
prior_cov = lx.DiagonalLinearOperator(4.0 * jnp.ones(2))  # (D_θ, D_θ) operator
# Λ_θ|y = Hᵀ Σ_ε⁻¹ H + Σ_θ⁻¹, a dense numpy reference
matrix = np.asarray(forward_map)
posterior_precision = matrix.T @ matrix / 0.1 + np.eye(2) / 4.0
# μ_θ|y = Λ_θ|y⁻¹ Hᵀ Σ_ε⁻¹ y
posterior_mean = np.linalg.solve(posterior_precision, matrix.T @ observation / 0.1)
# Σ_θ|y = Λ_θ|y⁻¹
posterior_cov = np.linalg.inv(posterior_precision)
print(posterior_mean.round(4), np.diag(posterior_cov).round(4))

for step_size, steps in ((1.0, 1), (0.5, 1), (0.5, 2), (0.5, 3)):
    # today (filterax)
    gnki = flx.processes.GNKI(
        scheduler=flx.FixedScheduler(dt=step_size),
        prior_mean=prior_mean,
        prior_cov=prior_cov,
    )
    end = iterate(gnki, members, observation, noise, forward, steps)
    # θ̄
    end_mean = einx.mean("[member] param", end.particles)  # → (D_θ,)
    # √(tr Σ̂_θ)
    # today (filterax)
    spread = jnp.sqrt(jnp.trace(flx.ensemble_covariance(end.particles).as_matrix()))
    error = jnp.abs(end_mean - posterior_mean).max()  # → ()
    print(step_size, steps, f"{error:.1e} {spread:.1e}")
```

It prints the posterior, `[ 1.2294 -1.1584] [0.0633 0.0589]`, and then one line per run with the error in the mean and $\sqrt{\op{tr}(\hat{\gmat{\Sigma}}_\theta)}$:

| $\Delta t$ | Steps | Mean error | Spread |
| --- | --- | --- | --- |
| 1 | 1 | `2.9e-07` | `6.8e-07` |
| 0.5 | 1 | `6.0e-01` | `1.5e+00` |
| 0.5 | 2 | `3.0e-01` | `7.3e-01` |
| 0.5 | 3 | `1.5e-01` | `3.7e-01` |

Both columns halve at each step, as [](#eq-processes-gnki-contraction) says, and at $\Delta t = 1$ the error of `2.9e-07` is what the jitter leaves.
The exact posterior has $\sqrt{\op{tr}(\gmat{\Sigma}_{\theta \mid y})} \approx 0.35$ here, which the spread passes through on its way to zero without stopping.

```{code-block} python
:caption: TEKI against the MAP. The error and the spread fall as $i^{-1/2}$.
:label: code-processes-teki
# Shapes: N_E = 200; D_θ = 2; D_y + D_θ = 5 augmented observations

# today (filterax)
teki = flx.processes.TEKI(
    scheduler=flx.FixedScheduler(dt=1.0), prior_mean=prior_mean, prior_cov=prior_cov
)
for steps in (1, 10, 100, 1000):
    end = iterate(teki, members, observation, noise, forward, steps)
    # θ̄
    end_mean = einx.mean("[member] param", end.particles)  # → (D_θ,)
    # √(tr Σ̂_θ)
    # today (filterax)
    spread = jnp.sqrt(jnp.trace(flx.ensemble_covariance(end.particles).as_matrix()))
    error = jnp.abs(end_mean - posterior_mean).max()  # → ()
    print(steps, f"{error:.5f} {spread:.5f}")
```

It prints:

| Iterations | Mean error | Spread |
| --- | --- | --- |
| 1 | `0.02008` | `0.04169` |
| 10 | `0.01764` | `0.03710` |
| 100 | `0.00974` | `0.02126` |
| 1000 | `0.00346` | `0.00768` |

The first step lands near the mode, since it is one Kalman update of the sample moments.
From 100 to 1000 iterations the two columns fall by 2.8, against the $\sqrt{10} \approx 3.2$ of the limit.
So the default budget of 200 iterations gives a usable mode, and more iterations buy little.

### EKS against the posterior

[](#code-processes-eks) runs the sampler on the same posterior with its default scheduler, and averages over the second half of the run.
A long run under a scan takes seconds, where the Layer 2 `run` is a Python loop.

```{code-block} python
:caption: EKS with the stable scheduler against the exact posterior, time-averaged after burn-in.
:label: code-processes-eks
# Shapes: N_iter = 500 iterations; N_E = 50 members; D_θ = 2


# user
# Θ⁽ⁱ⁾ at every iteration
def trajectory(process, start, target, noise_cov, model, steps):
    def body(state, _):
        evaluations = jax.vmap(model)(state.particles)  # → (N_E, D_y)
        # today (filterax)
        state = process.update(state, evaluations)
        return state, state.particles

    # today (filterax)
    initial = process.init(start, target, noise_cov)
    return jax.lax.scan(body, initial, None, length=steps)


# today (filterax)
sampler = flx.processes.EKS_Process(
    scheduler=flx.EKSStableScheduler(),
    key=jr.key(2),
    prior_mean=prior_mean,
    prior_cov=prior_cov,
)
fifty = members[:50]  # → (N_E, D_θ)
end, history = trajectory(sampler, fifty, observation, noise, forward, 500)
# θ⁽ᵉ⁾ pooled over the second half of the run
pooled = einx.rearrange("i e param -> (i e) param", history[250:])  # → (12500, D_θ)
sample_mean = einx.mean("[sample] param", pooled)  # → (D_θ,)
# today (filterax)
sample_cov = flx.ensemble_covariance(pooled).as_matrix()  # → (D_θ, D_θ)
print(sample_mean.round(4), np.diag(sample_cov).round(4))
print(f"{jnp.abs(sample_mean - posterior_mean).max():.4f}")
excess = np.trace(sample_cov) / np.trace(posterior_cov)
print(f"{excess:.3f} {end.algo_time:.2f}")


# user
# Δt from a state, for a stiff or a loose noise scale at the exact posterior
def settled_step(scale: float) -> Float[Array, ""]:
    precision = matrix.T @ matrix / scale + np.eye(2) / 4.0
    factor = np.linalg.cholesky(np.linalg.inv(precision))
    draws = np.random.default_rng(0).normal(size=(200, 2))
    draws = draws - draws.mean(0)
    # whiten so the sample covariance is exactly the identity
    draws = draws @ np.linalg.inv(np.linalg.cholesky(np.cov(draws.T))).T
    particles = jnp.asarray(draws @ factor.T)  # → (200, D_θ)
    # today (filterax)
    process_state = flx.ProcessState(
        particles=particles,
        forward_evals=jax.vmap(forward)(particles),
        obs=observation,
        noise_cov=lx.DiagonalLinearOperator(scale * jnp.ones(3)),
        step=jnp.asarray(0),
        algo_time=jnp.asarray(0.0),
        prior_cov=prior_cov,
    )
    # today (filterax)
    return flx.EKSStableScheduler().get_dt(process_state)


print(f"{settled_step(0.1):.4f} {settled_step(1e-4):.4f} {0.1 / np.sqrt(2.0):.4f}")
```

It prints `[ 1.2124 -1.1616] [0.0576 0.0596]`, then `0.0170`, then `0.959 34.17`, then `0.0707 0.0707 0.0707`.
The mean is within 0.017 of the exact posterior, and the total variance is 0.96 of the exact one, over a run reaching $\tau \approx 34$.
[](#eq-processes-eks-bias) predicts an excess of about 10% for a flat prior and a large ensemble.
This run has neither, so it does not test that number, and [](#code-processes-eks-step) does.
The last line is the scheduler at the exact posterior for noise variances of 0.1 and $10^{-4}$, next to $b / \sqrt{D_\theta}$.
The step does not depend on the scale of the problem, as [](#eq-processes-eks-stable) claims.

### A nonlinear problem against a grid

With $D_\theta = 2$ the posterior of a nonlinear model can be computed on a grid, which gives a reference for all of them at once.
The model is

$$h(\gvect{\theta}) = \begin{bmatrix} \theta_0 + 0.3 \theta_1^2 \\ \theta_1 + 0.3 \theta_0^2 \\ \theta_0 \theta_1 \end{bmatrix},$$ (eq-processes-curved)

with a standard normal prior and a noise variance of 0.1.

```{code-block} python
:caption: Each process on a nonlinear model, against the posterior on a 401 by 401 grid.
:label: code-processes-nonlinear
# Shapes: N_E = 200; D_θ = 2; D_y = 3; N_grid = 401² grid points


# user
# h(θ), as in the equation above
def curved(theta: Float[Array, " param"]) -> Float[Array, " obs"]:
    return jnp.stack(
        [
            theta[0] + 0.3 * theta[1] ** 2,
            theta[1] + 0.3 * theta[0] ** 2,
            theta[0] * theta[1],
        ]
    )


curved_target = curved(jnp.array([1.0, -0.5])) + jnp.array([0.15, -0.2, 0.1])
# today (lineax)
unit_prior = lx.DiagonalLinearOperator(jnp.ones(2))  # (D_θ, D_θ) operator
draws = jr.normal(jr.key(0), (200, 2))  # → (N_E, D_θ)

# a dense numpy reference: p(θ ∣ y) ∝ p(y ∣ θ) p(θ) on a grid
axis = np.linspace(-4.0, 4.0, 401)
grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
grid_mapped = np.asarray(jax.vmap(curved)(jnp.asarray(grid)))
log_density = -0.5 * ((np.asarray(curved_target) - grid_mapped) ** 2).sum(1) / 0.1
log_density = log_density - 0.5 * (grid**2).sum(1)
weights = np.exp(log_density - log_density.max())
weights = weights / weights.sum()
grid_mean = weights @ grid
grid_cov = (grid - grid_mean).T @ (weights[:, None] * (grid - grid_mean))
print(grid_mean.round(4), np.diag(grid_cov).round(4))


# user
def report(name: str, mean, covariance, count: int) -> None:
    error = np.abs(np.asarray(mean) - grid_mean).max()
    ratio = np.trace(np.asarray(covariance)) / np.trace(grid_cov)
    print(f"{name:12s} {error:.3f} {ratio:.3f} {count}")


twenty = flx.FixedScheduler(dt=0.05)
for name, process in (
    ("EKI", flx.processes.EKI(twenty)),
    ("ETKI", flx.processes.ETKI(twenty)),
):
    end = iterate(process, draws, curved_target, noise, curved, 20)
    # today (filterax)
    end_cov = flx.ensemble_covariance(end.particles).as_matrix()
    end_mean = einx.mean("[member] param", end.particles)  # → (D_θ,)
    report(name, end_mean, end_cov, 200 * 20)

# today (filterax)
uki_end = flx.UKI(
    forward_fn=curved, obs=curved_target, noise_cov=noise, scheduler=twenty
).run(jnp.zeros(2), unit_prior)
report("UKI", uki_end.mean, uki_end.covariance.as_matrix(), 5 * 20)

# today (filterax)
sampler = flx.processes.EKS_Process(
    scheduler=flx.EKSStableScheduler(),
    key=jr.key(2),
    prior_mean=jnp.zeros(2),
    prior_cov=unit_prior,
)
_, history = trajectory(sampler, draws[:50], curved_target, noise, curved, 500)
pooled = einx.rearrange("i e param -> (i e) param", history[250:])  # → (12500, D_θ)
# today (filterax)
pooled_cov = flx.ensemble_covariance(pooled).as_matrix()
pooled_mean = einx.mean("[sample] param", pooled)  # → (D_θ,)
report("EKS", pooled_mean, pooled_cov, 50 * 500)
```

It prints the grid posterior, `[ 0.8699 -0.5688] [0.0602 0.0596]`, and then:

| Process | Mean error | Variance over the grid's | Evaluations of $h$ |
| --- | --- | --- | --- |
| EKI, 20 steps | `0.068` | `0.623` | `4000` |
| ETKI, 20 steps | `0.065` | `1.198` | `4000` |
| UKI, 20 steps | `0.036` | `1.039` | `100` |
| EKS, 500 iterations, 50 members | `0.023` | `0.891` | `25000` |

The exactness of ETKI and UKI is gone, and what remains is a statistical linearisation.
Here UKI does much the best per evaluation, and EKS has the smallest error in the mean for the largest bill.
This nonlinearity is mild, and a stronger one may rank them differently.

## Numerics: float32, sigma points and gradients

ETKI's transform and UKI's standard form behave differently when the observations are strong.
UKI subtracts two nearly equal covariances in [](#eq-processes-uki), while ETKI works through the eigenvalues of $\gmat{\Lambda}_w$ ([chapter 6](06_etkf.md)).
[](#code-processes-float32) measures both against a float64 reference in information form, since the dense update of [](#code-processes-agreement) itself loses digits to the same subtraction.

```{code-block} python
:caption: Relative errors in float32 against an information-form reference, as the noise variance falls.
:label: code-processes-float32
# Shapes: N_E = 50; D_θ = 2; D_y = 3

small = 2.0 * jr.normal(jr.key(0), (50, 2))  # → (N_E, D_θ)
small_mean = np.asarray(einx.mean("[member] param", small))
small_cov = np.cov(np.asarray(small).T)
target = np.asarray(observation)

for variance in (0.1, 1e-3, 1e-6, 1e-8):
    # a dense float64 reference: Λ = Σ̂⁻¹ + Hᵀ H / σ²
    reference_precision = np.linalg.inv(small_cov) + matrix.T @ matrix / variance
    reference_cov = np.linalg.inv(reference_precision)
    right_side = np.linalg.solve(small_cov, small_mean) + matrix.T @ target / variance
    reference_mean = reference_cov @ right_side

    single_map = jnp.asarray(matrix, jnp.float32)  # → (D_y, D_θ)
    single_target = jnp.asarray(target, jnp.float32)  # → (D_y,)
    # today (lineax)
    single_noise = lx.DiagonalLinearOperator(jnp.full(3, variance, jnp.float32))
    single_members = jnp.asarray(small, jnp.float32)  # → (N_E, D_θ)
    single_forward = lambda theta: single_map @ theta

    # today (filterax)
    etki = flx.processes.ETKI(flx.FixedScheduler(dt=0.125))
    etki_end = iterate(
        etki, single_members, single_target, single_noise, single_forward, 8
    )
    etki_values = np.asarray(etki_end.particles, np.float64)

    # today (lineax)
    single_start = lx.MatrixLinearOperator(
        jnp.asarray(small_cov, jnp.float32), tags=lx.positive_semidefinite_tag
    )
    # today (filterax)
    uki = flx.UKI(
        forward_fn=single_forward,
        obs=single_target,
        noise_cov=single_noise,
        scheduler=flx.FixedScheduler(dt=0.125),
    ).run(jnp.asarray(small_mean, jnp.float32), single_start)

    scale_mean = np.abs(reference_mean).max()
    scale_cov = np.abs(reference_cov).max()
    etki_mean = etki_values.mean(0)
    etki_mean_error = np.abs(etki_mean - reference_mean).max() / scale_mean
    etki_cov_error = np.abs(np.cov(etki_values.T) - reference_cov).max() / scale_cov
    uki_mean = np.asarray(uki.mean, np.float64)
    uki_cov = np.asarray(uki.covariance.as_matrix(), np.float64)
    uki_mean_error = np.abs(uki_mean - reference_mean).max() / scale_mean
    uki_cov_error = np.abs(uki_cov - reference_cov).max() / scale_cov
    print(
        f"{variance:.0e} ETKI {etki_mean_error:.0e} {etki_cov_error:.0e}"
        f" UKI {uki_mean_error:.0e} {uki_cov_error:.0e}"
    )
```

It prints:

| $\sigma^2$ | ETKI mean | ETKI covariance | UKI mean | UKI covariance |
| --- | --- | --- | --- | --- |
| $10^{-1}$ | `9e-08` | `1e-06` | `5e-08` | `6e-07` |
| $10^{-3}$ | `1e-07` | `1e-06` | `2e-07` | `8e-06` |
| $10^{-6}$ | `1e-07` | `3e-05` | `2e-04` | `2e-02` |
| $10^{-8}$ | `4e-07` | `2e-04` | `1e-01` | `7e+02` |

With a prior variance of about 4, ordinary noise is fine in either form.
As the noise falls UKI loses its covariance first, and at $\sigma^2 = 10^{-8}$ that covariance is meaningless while ETKI still has three digits.
The pattern fits the subtraction, but that reading is an interpretation and not a separate test.
In float64 this is not a concern at these sizes, and for strong observations in float32 ETKI is the safer form.

The sigma-point weights matter more than the update does.
At $\alpha = 1$ the centre mean weight is zero, while at $\alpha = 10^{-3}$ it is about $-10^6$, and the moments come out of a cancellation between weights of that size.
gaussx warns about that in float32, and the UKI docstring's suggestion of a small $\alpha$ for a very nonlinear $h$ runs against that warning.

```{code-block} python
:caption: UKI on the nonlinear model at two values of $\alpha$, in both precisions.
:label: code-processes-alpha
import warnings

# Shapes: D_θ = 2; D_y = 3; 5 sigma points

for alpha in (1.0, 1e-3):
    for precision in (jnp.float64, jnp.float32):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # today (filterax)
            alpha_end = flx.UKI(
                forward_fn=curved,
                obs=jnp.asarray(curved_target, precision),
                noise_cov=lx.DiagonalLinearOperator(jnp.full(3, 0.1, precision)),
                scheduler=flx.FixedScheduler(dt=0.25),
                alpha=alpha,
            ).run(
                jnp.zeros(2, precision),
                lx.DiagonalLinearOperator(jnp.ones(2, precision)),
            )
        print(alpha, precision.__name__, np.asarray(alpha_end.mean).round(4))
```

It prints `[ 0.8628 -0.6138]` in both precisions at $\alpha = 1$, and at $\alpha = 10^{-3}$ it prints `[ 0.8697 -0.6216]` in float64 and `[ 0.8589 -0.6157]` in float32.
So $\alpha = 1$ loses nothing to float32, and $\alpha = 10^{-3}$ is wrong in the third digit.
The two values of $\alpha$ also differ from each other in float64, since they are different quadratures of a nonlinear $h$.

The steps themselves are differentiable, including the adaptive rule, because the `min` and `max` in [](#eq-processes-dmc) only choose between smooth branches.
So a Layer 1 loop can sit inside a gradient, which [](#code-processes-gradient) checks against a central difference for EKI under the adaptive rule.

```{code-block} python
:caption: The gradient of a Layer 1 run with respect to the noise variance, against a central difference.
:label: code-processes-gradient
# Shapes: N_E = 50; D_θ = 2; D_y = 3


# user
# 𝓛(σ²) = ‖θ̄‖², after running the process at Σ_ε = σ² I
def final_loss(process, variance: Float[Array, ""]) -> Float[Array, ""]:
    # today (lineax)
    scaled = lx.DiagonalLinearOperator(variance * jnp.ones(3))  # (D_y, D_y) operator
    end = iterate(process, small, curved_target, scaled, curved, 6)
    # θ̄
    end_mean = einx.mean("[member] param", end.particles)  # → (D_θ,)
    return jnp.sum(end_mean**2)  # → ()


at = jnp.asarray(0.1)
step = 1e-6
# today (filterax)
misfit_process = flx.processes.EKI(scheduler=flx.DataMisfitController())
gradient = jax.grad(lambda value: final_loss(misfit_process, value))(at)  # → ()
upper = final_loss(misfit_process, at + step)  # → ()
lower = final_loss(misfit_process, at - step)  # → ()
print(f"{gradient:.6f} {(upper - lower) / (2.0 * step):.6f}")
```

It prints `-0.565514 -0.565514`.

## When it breaks

**Past the stopping time, EKI fits the noise.**
On an ill-conditioned problem the least-squares fit is far from the truth.
An unregularised iteration heads for that fit once it has used up the useful part of the data.
This is why $\tau = 1$ is a stopping rule and not a formality.
[](#code-processes-semiconvergence) uses $D_\theta = D_y = 8$, with singular values from 1 down to about $3 \times 10^{-4}$ and a noise standard deviation of 0.01.

```{code-block} python
:caption: The semi-convergence of EKI, and TEKI holding at the MAP on the same problem.
:label: code-processes-semiconvergence
# Shapes: N_E = 100 members; D_θ = 8 parameters; D_y = 8 observations

left, _ = jnp.linalg.qr(jr.normal(jr.key(3), (8, 8)))
right, _ = jnp.linalg.qr(jr.normal(jr.key(4), (8, 8)))
# H = U diag(10^(−d / 2)) Vᵀ
singular = 10.0 ** (-jnp.arange(8) / 2.0)  # → (8,)
hard_map = einx.multiply("obs mode, mode -> obs mode", left, singular)
hard_map = einx.dot("obs mode, param mode -> obs param", hard_map, right)  # → (8, 8)
truth = jnp.ones(8)  # → (D_θ,)
hard_target = hard_map @ truth + 0.01 * jr.normal(jr.key(5), (8,))  # → (D_y,)
# today (lineax)
hard_noise = lx.DiagonalLinearOperator(1e-4 * jnp.ones(8))  # (D_y, D_y) operator
hard_members = jr.normal(jr.key(6), (100, 8))  # → (N_E, D_θ)


# user
def hard_forward(theta: Float[Array, " param"]) -> Float[Array, " obs"]:
    return hard_map @ theta


# today (filterax)
tenth = flx.FixedScheduler(dt=0.1)
hard_eki = flx.processes.EKI(tenth)
# today (filterax)
hard_teki = flx.processes.TEKI(
    scheduler=tenth,
    prior_mean=jnp.zeros(8),
    prior_cov=lx.DiagonalLinearOperator(jnp.ones(8)),
)
for steps in (10, 100, 1000, 10000):
    errors = []
    for process in (hard_eki, hard_teki):
        end = iterate(
            process, hard_members, hard_target, hard_noise, hard_forward, steps
        )
        # θ̄
        end_mean = einx.mean("[member] param", end.particles)  # → (D_θ,)
        errors.append(f"{jnp.linalg.norm(end_mean - truth):.2f}")
    print(steps, *errors)

# a dense reference: the fit and the MAP
dense_map = np.asarray(hard_map)
fit = np.linalg.solve(dense_map, np.asarray(hard_target))
hard_precision = dense_map.T @ dense_map / 1e-4 + np.eye(8)
hard_right = dense_map.T @ np.asarray(hard_target) / 1e-4
hard_mode = np.linalg.solve(hard_precision, hard_right)
print(f"{np.linalg.norm(fit - 1.0):.2f} {np.linalg.norm(hard_mode - 1.0):.2f}")
```

It prints:

| Iterations | $\tau$ | EKI error | TEKI error |
| --- | --- | --- | --- |
| 10 | 1 | `2.63` | `2.36` |
| 100 | 10 | `3.76` | `2.33` |
| 1000 | 100 | `7.80` | `2.32` |
| 10000 | 1000 | `13.05` | `2.31` |

and then `20.30 2.31`, the errors of the least-squares fit and of the MAP.
EKI at $\tau = 1$ has an error about that of the MAP, and it heads for the fit if allowed.
The Layer 2 `EKI` stops at $\tau = 1$ and so never sees this, while TEKI holds still because its cost has a minimum to hold on to.

**A large EKS step has no stationary law.**
[](#code-processes-eks-step) runs a scalar model with $\gamma = 1$ and 200 members and compares the variance with [](#eq-processes-eks-bias), then crosses the limit at $\Delta t = 1 / 2$.

```{code-block} python
:caption: The EKS variance against the formula below the limit, and its growth above it.
:label: code-processes-eks-step
# Shapes: N_iter = 1,000 iterations; N_E = 200 members; D_θ = D_y = 1

scalar_members = jr.normal(jr.key(1), (200, 1))  # → (N_E, D_θ)
scalar_target = jnp.zeros(1)  # → (D_y,)
# today (lineax)
scalar_noise = lx.DiagonalLinearOperator(jnp.ones(1))  # (D_y, D_y) operator


# user
def scalar_forward(theta: Float[Array, " param"]) -> Float[Array, " obs"]:
    return theta


for step_size in (0.1, 0.4, 0.6):
    # today (filterax)
    scalar_process = flx.processes.EKS_Process(
        scheduler=flx.FixedScheduler(dt=step_size), key=0
    )
    _, scalar_history = trajectory(
        scalar_process,
        scalar_members,
        scalar_target,
        scalar_noise,
        scalar_forward,
        1000,
    )
    settled = scalar_history[500:]  # → (500, N_E, D_θ)
    # θ̃⁽ᵉ⁾ at each iteration
    centred = settled - einx.mean("i [member] param", settled, keepdims=True)
    # Σ̂_θ, over the members and the iterations
    variance = jnp.sum(centred**2) / (500 * 199)  # → ()
    # ((1 − 2 Δt)^(−½) − 1) / Δt, for Δt < ½
    margin = 1.0 - 2.0 * step_size if step_size < 0.5 else float("nan")
    predicted = (margin**-0.5 - 1.0) / step_size
    print(step_size, f"{variance:.3g} {predicted:.3f}")
```

It prints `0.1 1.15 1.180`, `0.4 3.13 3.090` and `0.6 1.69e+76 nan`, with the measured variance before the formula's.
Below the limit the two agree within the sampling and the finite-ensemble pull, and a step of 0.4 triples the variance with no warning.
Above the limit nothing is stationary and the run diverges, which `EKSStableScheduler` exists to prevent.

**GNKI's jitter breaks a long run.**
$\delta$ in [](#eq-processes-gnki-gain) is absolute.
Once $\hat{\gmat{\Sigma}}_\theta$ falls below $\delta$, the estimate $\hat{\mat{J}}_h$ goes to zero, then $\mat{G}$ goes to zero, and [](#eq-processes-gnki) pulls every member back to the prior mean.
This is a bug rather than a limit of the method.
At $\Delta t = 1$ the second step already jumps to the prior mean, as [](#code-processes-gnki-jitter) shows.
Stop after the step that reaches $\tau = 1$, and do not run GNKI with a fixed step past that.

```{code-block} python
:caption: GNKI past its collapse, where the absolute jitter dominates $\hat{\gmat{\Sigma}}_\theta$.
:label: code-processes-gnki-jitter
# Shapes: N_E = 200; D_θ = 2; D_y = 3

for step_size, steps in ((1.0, 1), (1.0, 2), (0.5, 8), (0.5, 16)):
    # today (filterax)
    gnki = flx.processes.GNKI(
        scheduler=flx.FixedScheduler(dt=step_size),
        prior_mean=prior_mean,
        prior_cov=prior_cov,
    )
    end = iterate(gnki, members, observation, noise, forward, steps)
    # θ̄
    print(step_size, steps, einx.mean("[member] param", end.particles).round(4))
```

It prints `[ 1.2294 -1.1584]` after one step at $\Delta t = 1$, and `[ 0. -0.]` after two.
At $\Delta t = 0.5$ it prints `[ 1.2247 -1.1541]` after 8 steps, and `[0.4234 0.0199]` after 16.
The exact posterior mean is `[ 1.2294 -1.1584]`, so more iterations are moving away from the answer.

**SparseInversion shrinks everything if it runs on.**
The threshold in [](#eq-processes-sparse) does not fade with the spread.
Once the data term has stopped moving the members, the active parameters keep losing $\varrho \Delta t$ a step.
[](#code-processes-sparse) has two active parameters out of five.

```{code-block} python
:caption: SparseInversion over a short and a long run, with the true parameters $[2, 0, 0, -1.5, 0]$.
:label: code-processes-sparse
# Shapes: N_E = 100 members; D_θ = 5 parameters; D_y = 8 observations

sparse_map = jr.normal(jr.key(9), (8, 5))  # → (D_y, D_θ)
sparse_truth = jnp.array([2.0, 0.0, 0.0, -1.5, 0.0])  # → (D_θ,)
sparse_target = sparse_map @ sparse_truth + 0.05 * jr.normal(jr.key(10), (8,))
# today (lineax)
sparse_noise = lx.DiagonalLinearOperator(0.0025 * jnp.ones(8))  # (D_y, D_y) operator
sparse_members = jr.normal(jr.key(11), (100, 5))  # → (N_E, D_θ)


# user
def sparse_forward(theta: Float[Array, " param"]) -> Float[Array, " obs"]:
    return sparse_map @ theta


# today (filterax)
sparse = flx.processes.SparseInversion(
    scheduler=flx.FixedScheduler(dt=0.2), penalty_weight=0.05
)
for steps in (5, 100, 500):
    end = iterate(
        sparse, sparse_members, sparse_target, sparse_noise, sparse_forward, steps
    )
    # θ̄
    print(steps, einx.mean("[member] param", end.particles).round(3))
```

It prints `5 [ 1.96   0.     0.    -1.454  0.   ]`, `100 [ 1.252  0.     0.    -0.676  0.   ]` and `500 [0. 0. 0. 0. 0.]`.
At $\tau = 1$, after 5 steps, the support is right and the active values are close.
By 500 steps every parameter is zero, and an ensemble of zeros cannot move again, because it has no anomalies.

Three other limits are worth knowing:

- A small ensemble confines EKI, ETKI and TEKI to its span, so what the initial anomalies cannot express stays unreachable.
- A strongly nonlinear $h$ leaves ETKI and UKI as statistical linearisations, and their exactness above is a statement about linear models only.
- EKS has a bias from its step, and its members are correlated in time, so its average needs a burn-in and a long run.

## When each process is the right answer

| You want | Use | Cost in evaluations of $h$ |
| --- | --- | --- |
| A point estimate from a misfit, with no prior beyond the initial ensemble | `EKI` with `DataMisfitController` | $N_\mathcal{E}$ per step, and a handful of steps |
| The tempered posterior of the initial ensemble, with Gaussian moments | `processes.ETKI` with a schedule summing to 1 | $N_\mathcal{E}$ per step |
| A Gaussian belief with its covariance, at moderate $D_\theta$ | `UKI` | $2 D_\theta + 1$ per step |
| The posterior mean of a near-linear model, with $N_\mathcal{E} > D_\theta$ | `processes.GNKI`, stopping at $\tau = 1$ | $N_\mathcal{E}$ per step |
| The MAP under a Gaussian prior | `TEKI`, with patience | $N_\mathcal{E}$ per step, and many steps |
| Samples from a posterior that may not be Gaussian | `EKS` with `prior_mean` and `prior_cov` | $N_\mathcal{E}$ per step, and many steps |
| Parameters that should be exactly zero | `processes.SparseInversion`, stopped early | $N_\mathcal{E}$ per step |

To iterate on an initial ensemble rather than run these flows, [chapter 10](10_smoothers.md) has the iterative ensemble smoother, which needs no scheduler.

## See also

- [Chapter 4](04_kalman_update.md): the Kalman update and the cross-covariance form used here
- [Chapter 6](06_etkf.md): the transform behind ETKI, and its gradients
- [Chapter 10](10_smoothers.md): IES, an iteration anchored to the initial ensemble
- [Chapter 14](14_differentiable.md): differentiating through ensemble updates
- [Chapter 18](18_numerics.md): float32 and conditioning across the library
- [Chapter 20](20_inversion_example.md): an inversion from start to finish
- [API: processes](https://jejjohnson.github.io/filterax/reference/processes/), [schedulers](https://jejjohnson.github.io/filterax/reference/schedulers/) and [optax](https://jejjohnson.github.io/filterax/reference/optax/)
