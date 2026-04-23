"""State, result, and configuration containers.

All types are :class:`equinox.Module` subclasses — PyTree-compatible,
JIT-friendly, and safe to pass through ``jax.lax.scan`` or ``eqx.filter_jit``.

Iteration counters (``step``, ``algo_time``) are stored as scalar JAX array
leaves so they can be carried through ``lax.scan`` without triggering
recompilation; compile-time constants like ``n_ensemble`` and ``n_iterations``
stay in ``eqx.field(static=True)``.
"""

from __future__ import annotations

import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float, Int


class FilterState(eqx.Module, strict=True):
    """Running state for sequential filters."""

    particles: Float[Array, "N_e N_x"]
    step: Int[Array, ""]


class ProcessState(eqx.Module, strict=True):
    """Running state for iterative EKP processes."""

    particles: Float[Array, "J N_p"]
    forward_evals: Float[Array, "J N_d"]
    obs: Float[Array, " N_d"]
    noise_cov: lx.AbstractLinearOperator
    step: Int[Array, ""]
    algo_time: Float[Array, ""]


class UKIState(eqx.Module, strict=True):
    """Parametric state for Unscented Kalman Inversion.

    UKI tracks an explicit mean and covariance rather than particles.
    """

    mean: Float[Array, " N_p"]
    covariance: lx.AbstractLinearOperator
    step: Int[Array, ""]


class AnalysisResult(eqx.Module, strict=True):
    """Output of any analysis / update step."""

    particles: Float[Array, "N_e N_x"]
    log_likelihood: Float[Array, ""] | None = None
    diagnostics: dict | None = None


class FilterConfig(eqx.Module, strict=True):
    """Static configuration for sequential filters.

    ``localizer`` and ``inflator`` are forward-referenced through their
    abstract base classes to avoid a circular import with
    :mod:`filterax._src._protocols`.
    """

    n_ensemble: int = eqx.field(static=True)
    localizer: eqx.Module | None = None
    inflator: eqx.Module | None = None


class ProcessConfig(eqx.Module, strict=True):
    """Static configuration for EKP processes."""

    scheduler: eqx.Module
    n_iterations: int = eqx.field(static=True)
