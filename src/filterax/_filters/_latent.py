"""Latent-space ensemble filtering: codecs, lifted operators and latent dynamics.

Latent data assimilation composes the existing filters with a learned codec, an
encoder g_enc from the state x to a latent z of length D_z and a decoder g_dec back,
and adds no analysis step of its own. The ensemble algebra of the statistics, gain,
localization and inflation is unchanged; this module holds the wiring that runs a
deterministic square-root filter on the latent members z⁽ᵉ⁾ and reports the results
in state space.

The protocols are structural, so filterax never imports pipekit:

- an encoder is any callable ``x → z``, such as the ``.encode`` of a pipekit
  ``LatentMap``, and a decoder any callable ``z → x``;
- a latent map has both ``.encode`` and ``.decode``, `identity_latent_map` being the
  reference one;
- a latent forward model has ``.step(z, dt) → z``.

The functions `latent_ensemble`, `decode_ensemble` and `identity_latent_map` work on
whole ensembles, the modules `LatentDynamics`, `LiftedObs` and `EncodedDynamics` plug
into the filters, and the Layer-2 `filterax.LatentETKF` and `filterax.LatentLETKF`
live in `filterax._filters._models`.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from filterax._protocols import (
    AbstractDynamics,
    AbstractObsOperator,
)


def latent_ensemble(
    encoder: Any,
    x_ensemble: Float[Array, "N_e N_x"],
) -> Float[Array, "N_e N_z"]:
    """Encode every member into latent space.

        z⁽ᵉ⁾ = g_enc(x⁽ᵉ⁾)

    The encoder is mapped over the members with `equinox.filter_vmap`, so it may be a
    function, a bound ``.encode`` or an `equinox.Module`.

    Args:
        encoder: Encoder g_enc, applied to one state.
        x_ensemble: Ensemble x⁽ᵉ⁾ in state space, one row per member.

    Returns:
        The latent ensemble z⁽ᵉ⁾, one row per member.

    Examples:
        A linear encoder that keeps the first component:

        >>> import jax.numpy as jnp
        >>> import filterax
        >>> x_ensemble = jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        >>> filterax.latent_ensemble(lambda x: x[:1], x_ensemble).tolist()
        [[0.0], [2.0], [4.0]]

        The identity codec changes nothing:

        >>> codec = filterax.identity_latent_map(2)
        >>> latent = filterax.latent_ensemble(codec.encode, x_ensemble)
        >>> bool(jnp.array_equal(latent, x_ensemble))
        True
    """
    # z⁽ᵉ⁾ = g_enc(x⁽ᵉ⁾)
    return eqx.filter_vmap(encoder)(x_ensemble)


def decode_ensemble(
    decoder: Any,
    z_ensemble: Float[Array, "N_e N_z"],
) -> Float[Array, "N_e N_x"]:
    """Decode every member back into state space, the inverse of `latent_ensemble`.

        x⁽ᵉ⁾ = g_dec(z⁽ᵉ⁾)

    Args:
        decoder: Decoder g_dec, applied to one latent state.
        z_ensemble: Latent ensemble z⁽ᵉ⁾, one row per member.

    Returns:
        The ensemble x⁽ᵉ⁾ in state space, one row per member.

    Examples:
        A decoder that repeats the latent component:

        >>> import jax.numpy as jnp
        >>> import filterax
        >>> z_ensemble = jnp.array([[1.0], [2.0]])
        >>> decoder = lambda z: jnp.concatenate([z, z])
        >>> filterax.decode_ensemble(decoder, z_ensemble).tolist()
        [[1.0, 1.0], [2.0, 2.0]]
    """
    # x⁽ᵉ⁾ = g_dec(z⁽ᵉ⁾)
    return eqx.filter_vmap(decoder)(z_ensemble)


class IdentityLatentMap(eqx.Module, strict=True):
    """The identity codec, g_enc = g_dec = id, the reference for regression tests.

    With it, a latent filter must reproduce its state-space counterpart, which pins
    the latent wiring without a trained autoencoder.

    Attributes:
        dim: The shared length D_x = D_z, kept for shape checks.

    Examples:
        Both maps return their input:

        >>> import jax.numpy as jnp
        >>> import filterax
        >>> codec = filterax.IdentityLatentMap(dim=2)
        >>> codec.encode(jnp.array([1.0, 2.0])).tolist()
        [1.0, 2.0]
        >>> codec.decode(jnp.array([1.0, 2.0])).tolist()
        [1.0, 2.0]
    """

    dim: int = eqx.field(static=True)

    def encode(self, x: Float[Array, " N_x"]) -> Float[Array, " N_x"]:
        # z = g_enc(x) = x
        return x

    def decode(self, z: Float[Array, " N_x"]) -> Float[Array, " N_x"]:
        # x = g_dec(z) = z
        return z


def identity_latent_map(dim: int) -> IdentityLatentMap:
    """The identity codec of a given length, the reference for latent filters.

    The result has ``.encode`` and ``.decode``, so it satisfies the structural latent
    map protocol, and pins `filterax.LatentETKF` to the plain `filterax.ETKF` in the
    regression tests.

    Args:
        dim: The shared length D_x = D_z.

    Returns:
        An `IdentityLatentMap`.

    Examples:
        >>> import filterax
        >>> filterax.identity_latent_map(3)
        IdentityLatentMap(dim=3)
    """
    return IdentityLatentMap(dim=dim)


class LatentDynamics(AbstractDynamics, strict=True):
    """A raw latent forward model with ``.step(z, dt)``, as `AbstractDynamics`.

        z_t|t−1 = inner.step(z_t−1|t−1, t1 − t0)

    The wrapped object needs only the ``.step`` method, and its type is not checked.
    The wrapper lets latent dynamics drop into any Layer-2 forecast, which calls
    ``dynamics(state, t0, t1)``.

    Attributes:
        inner: The latent forward model, with ``.step(z, dt)``.

    Examples:
        A decay over the window length:

        >>> import jax.numpy as jnp
        >>> import filterax
        >>> class Decay:
        ...     def step(self, z, dt):
        ...         return z * (1.0 - dt)
        >>> dynamics = filterax.LatentDynamics(inner=Decay())
        >>> dynamics(jnp.array([2.0]), jnp.array(0.0), jnp.array(0.5)).tolist()
        [1.0]
    """

    inner: Any

    def __call__(
        self,
        state: Float[Array, " N_z"],
        t0: Float[Array, ""],
        t1: Float[Array, ""],
    ) -> Float[Array, " N_z"]:
        # z_t|t−1 = inner.step(z_t−1|t−1, t1 − t0)
        return self.inner.step(state, t1 - t0)


class LiftedObs(AbstractObsOperator, strict=True):
    """An observation operator on the state, lifted to the latent space.

        h_z(z) = h(g_dec(z))

    It is an `AbstractObsOperator` from the latent z to the observation y, so the
    filters and smoothers take it unchanged: they only see a map from z to y.

    Attributes:
        decoder: Codec whose ``.decode`` is the decoder g_dec.
        inner: Observation operator h on the state.

    Examples:
        Observe the first decoded component:

        >>> import jax.numpy as jnp
        >>> import filterax
        >>> lifted = filterax.LiftedObs(
        ...     decoder=filterax.identity_latent_map(2), inner=lambda x: x[:1]
        ... )
        >>> lifted(jnp.array([3.0, 4.0])).tolist()
        [3.0]
    """

    decoder: Any
    inner: AbstractObsOperator

    def __call__(self, state: Float[Array, " N_z"]) -> Float[Array, " N_y"]:
        # x = g_dec(z)
        decoded = self.decoder.decode(state)
        # y = h(x)
        return self.inner(decoded)


class EncodedDynamics(AbstractDynamics, strict=True):
    """State-space dynamics run through the codec, as dynamics on the latent.

        z_t|t−1 = g_enc(f(g_dec(z_t−1|t−1)))

    For a physical model or an ODE solver on the state when no learned latent dynamics
    exists: each step decodes, runs the dynamics f on the state, and encodes. Pass it
    as the ``dynamics`` of `filterax.LatentETKF`.

    Attributes:
        latent_map: Codec with ``.encode`` and ``.decode``, the encoder g_enc and the
            decoder g_dec.
        inner: Dynamics f on the state.

    Examples:
        With the identity codec it is the state-space dynamics:

        >>> import jax.numpy as jnp
        >>> import filterax
        >>> class Shift(filterax.AbstractDynamics):
        ...     def __call__(self, state, t0, t1):
        ...         return state + (t1 - t0)
        >>> dynamics = filterax.EncodedDynamics(
        ...     latent_map=filterax.identity_latent_map(1), inner=Shift()
        ... )
        >>> dynamics(jnp.array([1.0]), jnp.array(0.0), jnp.array(2.0)).tolist()
        [3.0]
    """

    latent_map: Any
    inner: AbstractDynamics

    def __call__(
        self,
        state: Float[Array, " N_z"],
        t0: Float[Array, ""],
        t1: Float[Array, ""],
    ) -> Float[Array, " N_z"]:
        # x = g_dec(z_t−1|t−1)
        x = self.latent_map.decode(state)
        # f(x)
        x_next = self.inner(x, t0, t1)
        # z_t|t−1 = g_enc(f(x))
        return self.latent_map.encode(x_next)


def _stack_decode(
    decoder: Any, z_history: Float[Array, "T N_e N_z"]
) -> Float[Array, "T N_e N_x"]:
    """Decode every member of a latent history, x⁽ᵉ⁾_t = g_dec(z⁽ᵉ⁾_t).

    A Python loop over the T windows keeps `decode_ensemble` mapping over the members
    only; T is small. An empty history still learns D_x by decoding one zero ensemble,
    so that its shape matches the non-empty case.

    Args:
        decoder: Decoder g_dec, applied to one latent state.
        z_history: Latent ensembles z⁽ᵉ⁾_t, stacked over the windows.

    Returns:
        The decoded ensembles x⁽ᵉ⁾_t, stacked over the windows.
    """
    T = z_history.shape[0]
    if T == 0:
        # decode one zero ensemble to learn D_x, so the empty result is (0, N_ℰ, D_x)
        N_e, N_z = z_history.shape[1], z_history.shape[2]
        probe = jnp.zeros((N_e, N_z), dtype=z_history.dtype)
        N_x = decode_ensemble(decoder, probe).shape[-1]
        return jnp.empty((0, N_e, N_x), dtype=z_history.dtype)
    # x⁽ᵉ⁾_t = g_dec(z⁽ᵉ⁾_t), window by window
    return jnp.stack([decode_ensemble(decoder, z_history[t]) for t in range(T)])
