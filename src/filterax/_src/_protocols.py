"""Protocol hierarchy for filterax.

Two protocol families match the library's two modes of operation:

* :class:`AbstractSequentialFilter` — one analysis update per assimilation
  window (EnKF, ETKF, EnSRF, LETKF, ...).
* :class:`AbstractProcess` — iterative updates until convergence (EKI, EKS,
  UKI, ...).

Shared extension points let users supply dynamics, observation operators,
noise models, localizers, inflators, and schedulers.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float, PRNGKeyArray

from filterax._src._types import AnalysisResult, ProcessState


class AbstractDynamics(eqx.Module, strict=True):
    r"""Forward model that propagates a single state from ``t0`` to ``t1``.

    Designed for use with :func:`equinox.filter_vmap` to broadcast over an
    ensemble::

        forecast = eqx.filter_vmap(dynamics)(particles, t0, t1)
    """

    @abc.abstractmethod
    def __call__(
        self,
        state: Float[Array, " N_x"],
        t0: Float[Array, ""],
        t1: Float[Array, ""],
    ) -> Float[Array, " N_x"]: ...


class AbstractObsOperator(eqx.Module, strict=True):
    """Map a single state vector from state space to observation space.

    Vectorise over an ensemble with :func:`equinox.filter_vmap`.
    """

    @abc.abstractmethod
    def __call__(
        self,
        state: Float[Array, " N_x"],
    ) -> Float[Array, " N_y"]: ...


class AbstractNoise(eqx.Module, strict=True):
    """Noise model backed by a gaussx / lineax covariance operator.

    Provides both the covariance operator (for Kalman-gain computations) and
    a sampling routine (for stochastic methods).
    """

    @abc.abstractmethod
    def covariance(self) -> lx.AbstractLinearOperator:
        """Return the covariance as a linear operator."""
        ...

    @abc.abstractmethod
    def sample(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
    ) -> Float[Array, "..."]:  # noqa: UP037
        """Draw noise samples of the given batch shape."""
        ...


class AbstractLocalizer(eqx.Module, strict=True):
    """Covariance localization strategy applied to the Kalman gain or covariance."""

    @abc.abstractmethod
    def __call__(
        self,
        cov: Float[Array, "N_x N_y"],
        coords: Any,
    ) -> Float[Array, "N_x N_y"]: ...


class AbstractInflator(eqx.Module, strict=True):
    """Ensemble inflation strategy applied after analysis."""

    @abc.abstractmethod
    def __call__(
        self,
        particles: Float[Array, "N_e N_x"],
    ) -> Float[Array, "N_e N_x"]: ...


class AbstractScheduler(eqx.Module, strict=True):
    """Step size / learning rate strategy for iterative processes."""

    @abc.abstractmethod
    def get_dt(self, state: ProcessState) -> Float[Array, ""]:
        """Return the step size for the current iteration."""
        ...


class AbstractSequentialFilter(eqx.Module, strict=True):
    """Protocol for sequential ensemble filters.

    Concrete filters implement a single ``analysis`` step; the forecast step
    is typically handled externally (``eqx.filter_vmap`` of dynamics over the
    ensemble) and passed back in as ``particles``.
    """

    @abc.abstractmethod
    def analysis(
        self,
        particles: Float[Array, "N_e N_x"],
        obs: Float[Array, " N_y"],
        obs_op: AbstractObsOperator
        | Callable[[Float[Array, " N_x"]], Float[Array, " N_y"]],
        obs_noise: lx.AbstractLinearOperator,
    ) -> AnalysisResult:
        """Assimilate an observation vector into the ensemble."""
        ...


class AbstractProcess(eqx.Module, strict=True):
    """Protocol for iterative ensemble Kalman processes.

    Concrete processes expose ``init`` and ``update``; the caller owns the
    outer iteration loop and supplies forward evaluations at each step.
    """

    @abc.abstractmethod
    def init(
        self,
        particles: Float[Array, "J N_p"],
        obs: Float[Array, " N_d"],
        noise_cov: lx.AbstractLinearOperator,
    ) -> ProcessState:
        """Initialise process state from a prior ensemble."""
        ...

    @abc.abstractmethod
    def update(
        self,
        state: ProcessState,
        forward_evals: Float[Array, "J N_d"],
    ) -> ProcessState:
        """Apply a single update step given forward evaluations."""
        ...
