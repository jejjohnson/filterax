# Latent-Space Ensemble DA

When $N_x$ is a high-resolution field — $10^6$ grid points of sea
surface height, say — even the ensemble trick strains: every member
must be propagated by an expensive model, the analysis manipulates
$(N_e, N_x)$ arrays, and the sample covariance lives in a space far
larger than the dynamics actually explore. If the system's effective
dynamics evolve on a low-dimensional manifold, a learned autoencoder
can expose it: assimilate in the latent space instead.

## The construction

Let $E: \mathbb{R}^{N_x} \to \mathbb{R}^{N_z}$ be an encoder and
$D: \mathbb{R}^{N_z} \to \mathbb{R}^{N_x}$ a decoder, with
$N_z \ll N_x$ and $D(E(x)) \approx x$ on the data manifold. The
latent ensemble is the encoded ensemble, member by member:

$$
z^{(j)} = E\big(x^{(j)}\big),
\qquad
Z \in \mathbb{R}^{N_e \times N_z}.
$$

The entire ensemble machinery — anomalies, gain, transform,
inflation — then runs unchanged on $Z$. Three pieces of wiring make
the cycle close:

**Lifted observation operator.** Observations live in $y$-space and
the observation operator $\mathcal{H}$ acts on $x$-space states, so
the latent filter observes through the composition

$$
\mathcal{H}_z = \mathcal{H} \circ D :
\quad z \;\xrightarrow{\;D\;}\; x \;\xrightarrow{\;\mathcal{H}\;}\; y.
$$

This is filterax's `LiftedObs`: it satisfies `AbstractObsOperator`,
so the ETKF/EnSRF/LETKF analysis only ever sees an $(N_z, N_y)$
interface and is oblivious to the decoder inside. The observation
error covariance $R$ operates in $y$-space and is untouched.
Importantly, $\mathcal{H} \circ D$ is nonlinear whenever $D$ is — but
ensemble filters never need a Jacobian; they linearise implicitly
through the ensemble.

**Latent dynamics.** Two options for the forecast:

- A learned latent surrogate $M_z: z_t \mapsto z_{t+1}$ — the cheap
  case, where the expensive physics is replaced by a small network
  trained alongside the codec (`LatentDynamics` wraps any object with
  `.step(z, dt)`).
- The original $x$-space model lifted through the codec round-trip,

  $$
  z_{t+1} = E\big( M\big( D(z_t) \big) \big)
  $$

  (`EncodedDynamics`) — no forecast speedup, but the *analysis* still
  runs in $\mathbb{R}^{N_z}$.

## When latent DA pays

- **High-dimensional fields with low intrinsic dimension.** The
  analysis cost terms in $N_x$ become terms in $N_z$; an
  $N_e = 50$ ensemble that is hopelessly rank-deficient in
  $\mathbb{R}^{10^6}$ can be a *generous* sample of
  $\mathbb{R}^{64}$.
- **Learned autoencoders already in the pipeline.** If a codec was
  trained for compression or emulation, assimilating in its latent
  space is nearly free — and the codec can be *fine-tuned through
  the filter*, since the whole construction is differentiable
  (chapter 14): gradients of the assimilation NLL flow into encoder
  and decoder weights.
- **Non-Gaussian state distributions.** A good encoder Gaussianises:
  the Kalman update's Gaussian assumption can hold better in $z$ than
  in $x$.

The caveat: analysis increments are confined to the decoder's range.
Whatever the codec cannot represent, the filter cannot correct. And
localization is problematic — latent dimensions of a typical
autoencoder are *global* modes with no intrinsic notion of physical
distance, which is why `LatentLETKF` in v0.1 deliberately applies
**no** localization in latent space (it is API parity for pipeline
wiring; its body matches `LatentETKF`).

## The two views: `LatentAssimilationResult`

The L2 wrappers surface both representations. The `_z` fields
(`particles_z`, `forecast_history_z`, `analysis_history_z`) are the
*primary* representation — the arrays the filter actually updated.
The $x$-space fields (`particles`, `forecast_history`,
`analysis_history`) are decoded views, provided so downstream
smoothers, diagnostics, and plots work unchanged. The
log-likelihoods are computed in observation space and are common to
both views. As always, axis 0 of each ensemble array is the member
axis and histories stack along a leading $T$ axis.

A useful regression anchor: with the identity codec
($E = D = \mathrm{id}$, `identity_latent_map`), `LatentETKF` must
reproduce plain `ETKF` bit-for-bit — the wrapper adds wiring, not
math.

## Implementation in filterax

Mirroring `tests/test_latent.py`: first the identity-codec parity
pin, then a dimension-reducing linear codec with a raw latent forward
model (auto-wrapped from its `.step(z, dt)` method):

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

import filterax as flx


class LinearObs(flx.AbstractObsOperator):
    H: jnp.ndarray

    def __call__(self, state):
        return self.H @ state


class IdentityDynamics(flx.AbstractDynamics):
    def __call__(self, state, t0, t1):
        return state


# Identity codec: LatentETKF must reproduce plain ETKF.
key = jax.random.key(0)
N_e, N_x, N_y = 30, 4, 2
particles = jax.random.normal(key, (N_e, N_x))
obs_op = LinearObs(H=jnp.eye(N_x)[:N_y])
R = lx.DiagonalLinearOperator(jnp.full((N_y,), 0.4))
obs_seq = [(jnp.array([0.3, -0.1]), 1.0), (jnp.array([0.2, 0.0]), 2.0)]

plain = flx.ETKF(dynamics=IdentityDynamics(), obs_op=obs_op).assimilate(
    particles, obs_seq, R
)
lm = flx.identity_latent_map(dim=N_x)
latent = flx.LatentETKF(
    latent_map=lm, dynamics=IdentityDynamics(), obs_op=obs_op
).assimilate(particles, obs_seq, R)

print("x-view matches ETKF:",
      bool(jnp.allclose(latent.particles, plain.particles, atol=1e-6)))
print("z-view == x-view under identity codec:",
      bool(jnp.allclose(latent.particles_z, latent.particles)))


# Dimension-reducing codec: assimilate in R^2 while observing R^4 fields.
class Codec(eqx.Module):
    W_enc: jnp.ndarray
    W_dec: jnp.ndarray

    def encode(self, x):
        return self.W_enc @ x

    def decode(self, z):
        return self.W_dec @ z


class LatentForward(eqx.Module):
    A: jnp.ndarray

    def step(self, z, dt):
        return self.A @ z


N_z = 2
k1, k2 = jax.random.split(jax.random.key(1))
codec = Codec(W_enc=jax.random.normal(k1, (N_z, N_x)),
              W_dec=jax.random.normal(k2, (N_x, N_z)))
fwd = LatentForward(A=0.9 * jnp.eye(N_z))   # raw .step(z, dt) — auto-wrapped

res = flx.LatentETKF(latent_map=codec, dynamics=fwd, obs_op=obs_op).assimilate(
    particles, obs_seq, R
)
print("x-space analysis:", res.analysis_history.shape)
print("z-space analysis:", res.analysis_history_z.shape)
print("log-likelihoods:", res.log_likelihoods.shape)
```

```
x-view matches ETKF: True
z-view == x-view under identity codec: True
x-space analysis: (2, 30, 4)
z-space analysis: (2, 30, 2)
log-likelihoods: (2,)
```

The second filter forecast and analysed a 2-dimensional latent
ensemble while observing (and reporting) 4-dimensional states; the
analysis arrays come back in both coordinate systems. With
`in_x_space=False`, `assimilate` accepts a pre-encoded $(N_e, N_z)$
ensemble and skips the initial encode.

A real workflow replaces `Codec` with a trained autoencoder (any
`eqx.Module` exposing `.encode`/`.decode` — structurally compatible
with `pipekit_cycle.LatentMap`) and `LatentForward` with a learned
latent surrogate; nothing else changes.

## Where next

- [Chapter 6 — ETKF](06_etkf.md): the analysis step running
  underneath, unchanged in $z$-space.
- [Chapter 14 — Differentiable assimilation](14_differentiable.md):
  training codec and latent dynamics through the filter.
- [Chapter 8 — LETKF](08_letkf.md): why R-localization needs physical
  coordinates — and hence why `LatentLETKF` skips it.
- [Chapter 16 — The assimilation cycle](16_assimilation_cycle.md):
  wiring latent filters into pipekit-cycle orchestration.
- [API: Advanced filters](../api/filters_advanced.md) — `LatentETKF`,
  `LatentLETKF`, `LiftedObs`, `LatentDynamics`, `EncodedDynamics`,
  `identity_latent_map`.

## References

- Peyron, M., Fillion, A., Gürol, S., Marchais, V., Gratton, S.,
  Boudier, P., & Goret, G. (2021). *Latent space data assimilation by
  using deep learning.* Q. J. R. Meteorol. Soc., 147(740),
  3759–3777.
- Amendola, M., Arcucci, R., Mottet, L., Casas, C. Q., Fan, S., Pain,
  C., Linden, P., & Guo, Y.-K. (2021). *Data assimilation in the
  latent space of a convolutional autoencoder.* ICCS 2021, LNCS
  12746.
- Chen, Y., Sanz-Alonso, D., & Willett, R. (2023). *Reduced-Order
  Autodifferentiable Ensemble Kalman Filters.* Inverse Problems,
  39(12), 124001. (Jointly learns the latent surrogate and decoder
  through the filter.)
