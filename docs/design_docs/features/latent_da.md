---
status: draft
version: 0.1.0
---

# filterax × Latent Data Assimilation

**Subject:** Ensemble Kalman filtering and smoothing in a learned
low-dimensional latent space. Adds two Layer-2 wrappers
(`LatentETKF`, `LatentLETKF`), a `LatentDynamics` / `LiftedObs`
component pair, and a `latent_ensemble` primitive. Built on the new
`pipekit_cycle.LatentMap` / `LatentForwardModel` protocols — no
pipekit import in the filterax core.

**Date:** 2026-05-28

**Decision anchor:** [D17 — Latent ensemble DA via composition + dedicated wrappers](../decisions.md#d17-latent-ensemble-da-via-composition--dedicated-wrappers).
Foundation in pipekit-cycle: `packages/pipekit-cycle/docs/design/latent.md`.

---

## 1  Motivation

Filterax today is **dimension-agnostic** by construction — the
ensemble's leading axis is `N_e` and the trailing axis is whatever the
user's `AbstractDynamics` produces. Nothing in the core math
requires that trailing axis to be a physical state. In principle a
user can already cobble together latent ensemble DA by writing two
adapters and supplying them in place of `AbstractDynamics` and
`AbstractObsOperator`.

What filterax is missing is:

1. **A canonical name for those adapters.** Every project re-implements
   `LatentDynamics` and `LiftedObs` with slight variations. The result
   is no shared vocabulary across vardax, filterax, plumax.
2. **Protocol-level cross-library compatibility.** vardax's
   `BilinAEPrior` exposes `encode` / `decode`; filterax users had to
   know that to wire it up. With `pipekit_cycle.LatentMap` runtime-checkable,
   the AE just *is* a LatentMap and filterax can compose with it
   structurally.
3. **A short path from a trained AE to a working filter.** Today the
   user writes ~50 lines of glue: vmap the encoder over the initial
   ensemble, wrap dynamics in encode-step-decode, wrap obs in
   decode-then-H, plumb the ensemble dimension through everything.
   `LatentETKF(latent_map=ae, dynamics=M_z, obs_op=H)` should do the
   same in one line.

This document ships those three things, all as **thin composition over
the existing Layer-1 components**. The ensemble math (statistics,
gain, localization, inflation) is unchanged.

---

## 2  Where it fits in the three-layer stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Layer 2 — Models                                                           │
│                                                                             │
│  Sequential filters:                                                        │
│    ETKF, EnSRF, StochasticEnKF, LETKF                                       │
│    + LatentETKF, LatentLETKF  (NEW — wrappers around ETKF/LETKF             │
│                               with LatentDynamics + LiftedObs prebound)     │
│                                                                             │
│  Smoothers:    EnKS, EnsembleRTS, FixedLagSmoother, IES                     │
│  Processes:    EKI, EKS, UKI                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 1 — Components                                                       │
│                                                                             │
│  Filters: ETKF, EnSRF, ESTKF, StochasticEnKF, LETKF analysis steps          │
│  Processes: EKI, EKS, UKI update steps                                      │
│  Localizers / Inflators / Schedulers / Noise (unchanged)                    │
│                                                                             │
│  Latent (NEW):                                                              │
│    LatentDynamics(AbstractDynamics)  — wraps LatentForwardModel             │
│    LiftedObs(AbstractObsOperator)    — wraps (Decoder, ObservationOperator) │
│    EncodedDynamics(AbstractDynamics) — round-trips an x-dynamics via AE     │
├─────────────────────────────────────────────────────────────────────────────┤
│  Layer 0 — Primitives                                                       │
│                                                                             │
│  Ensemble algebra: mean, anomalies, covariance, cross-covariance, gain      │
│  Localization, inflation, perturbations (unchanged)                         │
│                                                                             │
│  + latent_ensemble(encoder, x_ensemble)  — vmap(encode) helper              │
│  + decode_ensemble(decoder, z_ensemble)  — vmap(decode) helper              │
│  + identity_latent_map(N)                — regression test fixture          │
└─────────────────────────────────────────────────────────────────────────────┘
```

No primitive is removed; no signature changes. The Layer-1 additions
are concrete `AbstractDynamics` / `AbstractObsOperator` subclasses that
expect to consume `pipekit_cycle.LatentMap` instances structurally.

---

## 3  The math, in one screen

Let $\{x^{(i)}\}_{i=1}^{N_e} \subset \mathbb{R}^{N_x}$ be the ensemble
in physical space and $\{z^{(i)}\}_{i=1}^{N_e} \subset \mathbb{R}^{N_z}$
the ensemble in latent space. The relationships:

$$
z^{(i)} = \varphi(x^{(i)}), \qquad
\hat{x}^{(i)} = \psi(z^{(i)}), \qquad
\tilde{H}(z) := H(\psi(z)).
$$

Empirical statistics in $\mathcal{Z}$:

$$
\bar{z} = \tfrac{1}{N_e}\sum_i z^{(i)}, \qquad
Z' = Z - \mathbf{1}\bar{z}^\top, \qquad
\mathbf{P}_z = \tfrac{1}{N_e - 1} Z'^\top Z'.
$$

Cross-covariance with predicted observations $\hat{y}^{(i)} =
\tilde{H}(z^{(i)})$:

$$
\mathbf{C}_{z y} = \tfrac{1}{N_e - 1} Z'^\top \hat{Y}'.
$$

Kalman gain in $\mathcal{Z}$:

$$
\mathbf{K}_z = \mathbf{C}_{z y} \bigl(\hat{Y}'^\top \hat{Y}' / (N_e - 1) + \mathbf{R}\bigr)^{-1}.
$$

Ensemble update:

$$
z^{(i)} \;\leftarrow\; z^{(i)} + \mathbf{K}_z \bigl(y - \tilde{H}(z^{(i)}) + \varepsilon^{(i)}\bigr), \qquad \varepsilon^{(i)} \sim \mathcal{N}(0, \mathbf{R}) \text{ (stochastic forms only)}.
$$

When we read off the analysis in physical space we decode:
$\hat{x}^{(i)} = \psi(z^{(i)})$.

**Structural observation.** None of this is filterax-specific math —
it is the same gain formula as ETKF/EnSRF/LETKF, applied to a
different trailing axis. That is precisely why no algorithmic change
is needed inside the analysis steps. The wrappers exist for ergonomics
and for compile-time correctness (so the user can't accidentally feed a
z-space ensemble into an x-space analysis step).

---

## 4  New Layer-1 components

All live in a new module `filterax/_src/latent.py`. They consume
`pipekit_cycle.LatentMap` and `pipekit_cycle.LatentForwardModel`
structurally — filterax does not import pipekit-cycle (D11 still
holds; the import is in optional `tests/test_latent_pipekit_compat.py`).

### 4.1  `LatentDynamics`

```python
class LatentDynamics(AbstractDynamics):
    """Wrap a LatentForwardModel as an AbstractDynamics.

    The ensemble that flows through this dynamics is in z-space
    (shape ``(N_e, N_z)``).  vmap is applied externally as for any
    AbstractDynamics.
    """

    inner: Any                                # LatentForwardModel (structural)

    def __call__(self, latent, t0, t1):
        dt = t1 - t0
        return self.inner.step(latent, dt)
```

### 4.2  `LiftedObs`

```python
class LiftedObs(AbstractObsOperator):
    """Compose a Decoder with an x-space ObservationOperator.

    Maps  z ↦ y  via  z ─psi→ x ─H→ y.  Satisfies AbstractObsOperator
    so it slots into ETKF, EnSRF, LETKF, EnKS without changes.

    The ``decoder`` field is expected to satisfy
    ``pipekit_cycle.Decoder`` — i.e. expose a ``.decode(z)`` method.
    A full ``LatentMap`` works directly because it is a subtype of
    ``Decoder``; a bare encoder/decoder pair likewise.
    """

    decoder: Any                              # Decoder (structural; has .decode)
    inner: AbstractObsOperator                # x-space H

    def __call__(self, latent):
        return self.inner(self.decoder.decode(latent))
```

### 4.3  `EncodedDynamics`

```python
class EncodedDynamics(AbstractDynamics):
    """Lift an x-dynamics into z-space via the AE round-trip.

    For users who have an x-space ``AbstractDynamics`` (often a wrapped
    physics model or `diffrax` ODE) but no learned M_z.  One ensemble
    step costs encode + dynamics + decode, vmapped over members.
    """

    latent_map: Any                           # LatentMap (structural)
    inner: AbstractDynamics

    def __call__(self, latent, t0, t1):
        x = self.latent_map.decode(latent)
        x_next = self.inner(x, t0, t1)
        return self.latent_map.encode(x_next)
```

---

## 5  New Layer-0 primitives

Two helpers and one test fixture in `filterax/_src/latent.py`:

```python
def latent_ensemble(encoder, x_ensemble):
    """Encode each member.  (N_e, N_x) → (N_e, N_z)."""
    return eqx.filter_vmap(encoder)(x_ensemble)

def decode_ensemble(decoder, z_ensemble):
    """Decode each member.  (N_e, N_z) → (N_e, N_x)."""
    return eqx.filter_vmap(decoder)(z_ensemble)

def identity_latent_map(N: int):
    """phi = psi = identity.  Reduces LatentETKF to ETKF for regression tests."""
    ...
```

---

## 6  Layer-2 wrappers

The headline ergonomics. Two new classes in `filterax/_src/models.py`,
paralleling the existing `ETKF` / `LETKF` wrappers.

### 6.1  `LatentETKF`

```python
class LatentETKF(eqx.Module):
    """ETKF analysis applied to a z-space ensemble.

    Construction is one line if you have an AE and a latent dynamics::

        flx.LatentETKF(
            latent_map=ae,                 # pipekit_cycle.LatentMap
            dynamics=M_z,                  # LatentForwardModel (or EncodedDynamics)
            obs_op=H_x,                    # x-space ObservationOperator
            inflator=flx.RTPS(alpha=0.9),
        )
    """

    latent_map: Any
    dynamics: AbstractDynamics                # already in z-space
    obs_op: AbstractObsOperator               # x-space H; lifted internally
    inflator: AbstractInflator | None = None
    _filter: AbstractSequentialFilter = field(init=False)
    _lifted_H: AbstractObsOperator = field(init=False)

    def __post_init__(self):
        self._filter = ETKF()
        self._lifted_H = LiftedObs(decoder=self.latent_map, inner=self.obs_op)

    def assimilate(
        self, init_ensemble, observations, obs_noise, *, in_x_space=True
    ) -> AssimilationResult:
        """Run a full forecast-analysis loop.

        Args:
            init_ensemble: (N_e, N_x) or (N_e, N_z).  Pass ``in_x_space=True``
                (default) for x-space; we encode once.
            observations: list of (y_t, t).
            obs_noise: AbstractNoise in y-space.
        """
        z0 = (latent_ensemble(self.latent_map.encode, init_ensemble)
              if in_x_space else init_ensemble)

        analyses_z, log_lik_history = _scan_filter(
            filter_=self._filter,
            dynamics=self.dynamics,
            obs_op=self._lifted_H,
            ensemble=z0, observations=observations, obs_noise=obs_noise,
            inflator=self.inflator,
        )
        analyses_x = decode_ensemble(self.latent_map.decode, analyses_z)

        return AssimilationResult(
            particles=analyses_x,
            particles_z=analyses_z,
            log_likelihoods=log_lik_history,
        )
```

### 6.2  `LatentLETKF`

Identical wrapping pattern over `LETKF`. The localizer operates in
$\mathcal{Z}$ — see §9.2 for the caveat.

```python
class LatentLETKF(eqx.Module):
    latent_map: Any
    dynamics: AbstractDynamics
    obs_op: AbstractObsOperator
    localizer: AbstractLocalizer | None = None
    inflator: AbstractInflator | None = None
    ...
```

---

## 7  End-to-end usage

### 7.1  Lorenz-96, $N_z = 8$, learned $M_z$

```python
import filterax as flx
import pipekit_cycle as pc
import equinox as eqx
import jax.numpy as jnp

# 1. Pretrained AE (any eqx.Module exposing .encode/.decode/.latent_dim/.state_signature).
ae = MyBilinearAE(state_dim=40, latent_dim=8)
assert isinstance(ae, pc.LatentMap)          # structural check

# 2. Latent dynamics — learned residual MLP, satisfies LatentForwardModel.
M_z = MyLearnedLatentDynamics(latent_dim=8, dt=0.01)

# 3. Construct the filter.
filt = flx.LatentETKF(
    latent_map=ae,
    dynamics=flx.LatentDynamics(inner=M_z),
    obs_op=flx.IdentityObs(),                # x-space identity
    inflator=flx.RTPS(alpha=0.9),
)

# 4. Run.
x0_ens = jax.random.normal(key, (50, 40))    # 50 members in x-space
res = filt.assimilate(
    init_ensemble=x0_ens,
    observations=[(y_t1, t1), (y_t2, t2), ...],
    obs_noise=R,
)
# res.particles    -> (T, N_e, N_x)
# res.particles_z  -> (T, N_e, N_z)
```

### 7.2  No learned $M_z$ — use the AE round-trip

```python
# Same AE, but no latent dynamics yet.  Wrap the physics model:
physics = MyOdeDynamics(...)                 # AbstractDynamics in x-space

filt = flx.LatentETKF(
    latent_map=ae,
    dynamics=flx.EncodedDynamics(latent_map=ae, inner=physics),
    obs_op=flx.IdentityObs(),
)
# Otherwise identical.  Slower per step (encode+decode each forecast),
# but no M_z to train.
```

### 7.3  Cross-library — vardax AE + filterax filter

```python
import vardax as vdx
import filterax as flx

ae = vdx.BilinAEPrior1D(state_dim=40, latent_dim=8, ...)
assert isinstance(ae, pc.LatentMap)          # works — vardax priors satisfy the protocol

filt = flx.LatentETKF(latent_map=ae, dynamics=..., obs_op=...)
```

This is the **point** of putting the protocol in pipekit-cycle:
filterax never imports vardax, and vardax never imports filterax, but
they compose at the protocol level.

---

## 8  Differentiability

Latent filters are differentiable for exactly the same reasons the
x-space filters are (see [features/differentiable_da.md](differentiable_da.md)):

* The ensemble math (`ensemble_mean`, `ensemble_covariance`,
  `kalman_gain`) is smooth.
* `eqx.Module` AEs are autodiff-friendly.
* `jax.lax.scan` + `jax.checkpoint` work unchanged on the latent
  ensemble.

Gradients can flow through:

| Quantity | Path |
|---|---|
| AE weights (φ, ψ) | through encode of init ensemble, through decode in `LiftedObs` |
| Latent dynamics $M_z$ | through forecast scan |
| Observation operator $H$ | through `LiftedObs.inner` |
| Filter hyperparameters | inflation, localization radius — same as x-space |

A natural training loss is the observation-space NLL accumulated over a
window. The cost is computed by `_scan_filter` and returned as
`log_likelihoods`; users wrap it in their training loop.

**Memory note.** Reverse-mode AD stores the *latent* ensemble per
step, not the x-space ensemble. For $N_z = 8$ vs $N_x = 10^4$ the
tape shrinks by three orders of magnitude. This is the second
headline win of latent ensemble DA, after the gain solve cost.

---

## 9  Localization, inflation, and other extension points

### 9.1  Inflation

Multiplicative, RTPS, RTPP, and additive inflation all operate on the
ensemble's leading axis. They are dimension-agnostic and apply to z
ensembles without change.

```python
flx.RTPS(alpha=0.9)                 # applied to z ensemble; same code
```

Additive inflation in $\mathcal{Z}$ has the same stochasticity caveat
as in $\mathcal{X}$ — see differentiable_da.md §5.5.

### 9.2  Localization

Localization is **physically meaningful in $\mathcal{X}$**, not in
$\mathcal{Z}$. The latent dimensions of a typical AE are global modes,
not local features. Three options:

(a) **No localization.** Default for `LatentETKF`; usually fine because
$N_z \ll N_e$ kills the sampling noise that motivates localization.

(b) **Localize the lifted gain in $\mathcal{X}$.** Compute $\mathbf{K}_z$
in $\mathcal{Z}$, decode to $\mathbf{K}_x$, taper there. Requires
materialising $\psi'$ — expensive. Possible but not in v0.1.

(c) **Localize per-latent-mode.** Some AEs (e.g., U-Net) have
modes with implicit spatial support. `LatentLETKF` accepts an
optional `mode_coords` that gives each latent dimension a nominal
spatial coordinate; the Gaspari–Cohn taper uses these coordinates.
This is a research-grade feature, marked experimental in v0.1.

For v0.1 we ship (a) as the default and document (b) and (c) as
follow-up work.

### 9.3  Smoothers

`EnKS`, `EnsembleRTS`, `FixedLagSmoother`, `IES` all operate on
ensembles; they compose with `LiftedObs` and `LatentDynamics` exactly
like the filters. We add `LatentEnKS` only if there is demonstrated
demand — composition is short enough for users to write inline.

---

## 10  Decision D17 (filed in `decisions.md`)

> Latent ensemble DA is achieved in filterax by **composition over the
> existing Layer-1 components**, not by new analysis-step classes.
> Filterax adds:
> * Three concrete Layer-1 components — `LatentDynamics`, `LiftedObs`,
>   `EncodedDynamics` — that consume `pipekit_cycle.LatentMap` /
>   `LatentForwardModel` structurally.
> * Two Layer-2 wrappers — `LatentETKF`, `LatentLETKF` — that pre-bind
>   those components onto the existing ETKF / LETKF analysis steps.
> * Two Layer-0 helpers — `latent_ensemble`, `decode_ensemble`.
>
> Filterax does **not** introduce a parallel `Abstract*LatentFilter`
> protocol family. The existing `AbstractSequentialFilter` already
> works on z-space ensembles; latency comes from ergonomics, not from
> the math.
>
> D11 (structural pipekit-cycle compatibility) extends transparently:
> the new `pipekit_cycle.LatentMap` / `LatentForwardModel` protocols
> are consumed structurally; filterax imports nothing from pipekit.

---

## 11  Acceptance criteria for v0.1

* `LatentDynamics`, `LiftedObs`, `EncodedDynamics`, `LatentETKF`,
  `LatentLETKF`, `latent_ensemble`, `decode_ensemble`,
  `identity_latent_map` importable from `filterax`.
* Identity-AE regression test: with `latent_map =
  identity_latent_map(40)` on Lorenz-96, `LatentETKF` matches `ETKF`
  bit-for-bit (deterministic) and within $1e{-5}$ stat distance
  (stochastic forms).
* Cross-library structural test (optional, gated on `pipekit_cycle`):
  `isinstance(vardax_ae, pc.LatentMap)` and the resulting
  `LatentETKF` runs end-to-end on a fixture.
* Differentiability test: `jax.grad` w.r.t. AE weights over a
  three-step assimilation; gradient is finite and matches finite
  differences.
* Notebook: Lorenz-96, $N_z = 8$, side-by-side `ETKF` vs `LatentETKF`
  with `EncodedDynamics` (no learned $M_z$).
* No regression in existing ETKF / EnSRF / LETKF / EnKS tests.

---

## 12  Out of scope (deferred)

* **`LatentEnKS` smoother class.** Compose inline for now; promote to
  Layer-2 only on demand.
* **Localization in $\mathcal{X}$ via $\psi'$ pushforward.** Useful
  but $O(N_x \cdot N_z)$ memory; revisit when a concrete operational
  case appears.
* **VAE-aware filters.** Stochastic encoders require a `sample_encode`
  protocol method, deferred to `pipekit_cycle.latent` v0.2.
* **Patcher-LETKF in latent space.** D16 (patcher-LETKF) lands in
  x-space first; latent patcher composition is straightforward once
  D16 is in.

---

## 13  References

1. Peyron, M. et al. (2021). *Latent space data assimilation by using
   deep learning.* QJRMS.
2. Cheng, S. et al. (2023). *Generalised latent assimilation in
   heterogeneous reduced spaces with machine learning surrogates.*
   J. Sci. Comput.
3. Mack, J., Arcucci, R., Molina-Solana, M. & Guo, Y.-K. (2020).
   *Attention-based convolutional autoencoders for 3D-variational data
   assimilation.* Comput. Methods Appl. Mech. Eng.
4. Bocquet, M., Brajard, J., Carrassi, A. & Bertino, L. (2019). *Data
   assimilation as a learning tool to infer ODE representations of
   dynamical models.* NPG.
5. filterax differentiable DA — [features/differentiable_da.md](differentiable_da.md).
6. pipekit-cycle foundation — `packages/pipekit-cycle/docs/design/latent.md`.
