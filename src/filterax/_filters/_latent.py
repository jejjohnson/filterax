"""Latent-space ensemble Kalman filtering (Wave 5.C+).

Implements decision D17 — latent ensemble DA via thin composition over
the existing Layer-1 components, with no new analysis-step classes.
The ensemble math (statistics, gain, localization, inflation) is
unchanged; this module adds the wiring to run any deterministic
square-root filter in a learned latent space ``ℝ^{N_z}`` and surface
results back in ``ℝ^{N_x}``.

Protocol expectations (duck-typed; no pipekit import per D11)
-------------------------------------------------------------
* **Encoder** — anything callable as ``encoder(x: Float[Array, " N_x"])
  → Float[Array, " N_z"]`` (typically the ``.encode`` method on a
  ``pipekit_cycle.LatentMap``).
* **Decoder** — same shape for ``decoder(z) → x``.
* **LatentMap** — an object exposing both ``.encode`` and ``.decode``.
  :func:`identity_latent_map` is the canonical fixture; user
  autoencoders satisfy this structurally.
* **LatentForwardModel** — an object exposing ``.step(z, dt) → z``.

Layer
-----
* Layer 0: :func:`latent_ensemble`, :func:`decode_ensemble`,
  :func:`identity_latent_map`.
* Layer 1: :class:`LatentDynamics`, :class:`LiftedObs`,
  :class:`EncodedDynamics`.
* Layer 2: :class:`filterax.LatentETKF`, :class:`filterax.LatentLETKF`
  (in :mod:`filterax._filters._models`).
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
    """Encode each ensemble member into latent space.

    ``vmap``s the encoder over the ensemble leading axis. The encoder
    is anything callable as ``encoder(x) → z`` — typically the
    ``.encode`` method bound off a :class:`pipekit_cycle.LatentMap` or
    a bare encoder ``eqx.Module``.

    Args:
        encoder: Callable ``x → z``.
        x_ensemble: ``(Nₑ, Nₓ)`` ensemble in state space.

    Returns:
        ``(Nₑ, N_z)`` ensemble in latent space.
    """
    return eqx.filter_vmap(encoder)(x_ensemble)


def decode_ensemble(
    decoder: Any,
    z_ensemble: Float[Array, "N_e N_z"],
) -> Float[Array, "N_e N_x"]:
    """Decode each ensemble member back into state space.

    Mirror of :func:`latent_ensemble` for the decode direction.

    Args:
        decoder: Callable ``z → x``.
        z_ensemble: ``(Nₑ, N_z)`` ensemble in latent space.

    Returns:
        ``(Nₑ, Nₓ)`` ensemble in state space.
    """
    return eqx.filter_vmap(decoder)(z_ensemble)


class IdentityLatentMap(eqx.Module, strict=True):
    r"""``φ = ψ = id`` regression-test fixture.

    Used to verify that the latent filters reduce to their state-space
    counterparts when the codec is the identity — pinning the wrapping
    layer without depending on a real autoencoder. Carries a single
    static ``dim`` field so callers can sanity-check shapes.
    """

    dim: int = eqx.field(static=True)

    def encode(self, x: Float[Array, " N_x"]) -> Float[Array, " N_x"]:
        return x

    def decode(self, z: Float[Array, " N_x"]) -> Float[Array, " N_x"]:
        return z


def identity_latent_map(dim: int) -> IdentityLatentMap:
    """Construct an :class:`IdentityLatentMap` of the given dimension.

    The returned object satisfies the structural ``LatentMap`` contract
    (``.encode`` + ``.decode``) and is the canonical fixture for
    regression tests that pin :class:`filterax.LatentETKF` to plain
    :class:`filterax.ETKF`.

    Args:
        dim: Common state / latent dimension.

    Returns:
        Pass-through latent map.
    """
    return IdentityLatentMap(dim=dim)


class LatentDynamics(AbstractDynamics, strict=True):
    r"""Wrap a structural ``LatentForwardModel`` as an ``AbstractDynamics``.

    The wrapped object only needs a ``.step(z, dt)`` method; nothing
    else about its type is constrained (no ``isinstance`` check). The
    wrapper exists so the latent dynamics can drop into any existing
    Layer-2 forecast loop alongside other ``AbstractDynamics``
    instances.

    Attributes:
        inner: The wrapped latent forward model.
    """

    inner: Any

    def __call__(
        self,
        state: Float[Array, " N_z"],
        t0: Float[Array, ""],
        t1: Float[Array, ""],
    ) -> Float[Array, " N_z"]:
        return self.inner.step(state, t1 - t0)


class LiftedObs(AbstractObsOperator, strict=True):
    r"""Compose a decoder with an ``x``-space observation operator.

    Maps ``z → y`` via ``z ─ψ→ x ─H→ y``. Satisfies
    :class:`AbstractObsOperator`, so it slots into ETKF / EnSRF /
    LETKF / EnKS without changes — they only ever see the
    ``(N_z, N_y)`` interface and don't care what's inside.

    Attributes:
        decoder: Anything callable as ``decoder.decode(z) → x``.
        inner: Standard ``x``-space observation operator.
    """

    decoder: Any
    inner: AbstractObsOperator

    def __call__(self, state: Float[Array, " N_z"]) -> Float[Array, " N_y"]:
        return self.inner(self.decoder.decode(state))


class EncodedDynamics(AbstractDynamics, strict=True):
    r"""Lift an ``x``-space dynamics into ``z``-space via codec round-trip.

    For users with an ``x``-space :class:`AbstractDynamics` (a physics
    model or a ``diffrax`` ODE wrapper) but no learned ``M_z``. One
    ensemble step costs ``decode → inner → encode``. Used as the
    ``dynamics`` argument to :class:`filterax.LatentETKF` when no
    latent dynamics is available.

    Attributes:
        latent_map: ``.encode`` / ``.decode`` codec.
        inner: ``x``-space dynamics.
    """

    latent_map: Any
    inner: AbstractDynamics

    def __call__(
        self,
        state: Float[Array, " N_z"],
        t0: Float[Array, ""],
        t1: Float[Array, ""],
    ) -> Float[Array, " N_z"]:
        x = self.latent_map.decode(state)
        x_next = self.inner(x, t0, t1)
        return self.latent_map.encode(x_next)


def _stack_decode(
    decoder: Any, z_history: Float[Array, "T N_e N_z"]
) -> Float[Array, "T N_e N_x"]:
    """Decode every member of a ``(T, Nₑ, N_z)`` history into ``x``-space.

    Used internally by the Layer-2 latent wrappers to surface the
    latent ``analysis_history`` / ``forecast_history`` in state space
    while keeping the latent representations on the side. Implemented
    as a Python ``for`` over the time axis so the inner ``decode_ensemble``
    can keep its ensemble-axis ``vmap`` semantics; ``T`` is small.
    """
    T = z_history.shape[0]
    if T == 0:
        # Probe the decoder once to learn ``N_x`` and return an empty
        # ``(0, N_e, N_x)`` so the result shape stays consistent with the
        # non-empty case.
        N_e, N_z = z_history.shape[1], z_history.shape[2]
        probe = jnp.zeros((N_e, N_z), dtype=z_history.dtype)
        N_x = decode_ensemble(decoder, probe).shape[-1]
        return jnp.empty((0, N_e, N_x), dtype=z_history.dtype)
    return jnp.stack([decode_ensemble(decoder, z_history[t]) for t in range(T)])
