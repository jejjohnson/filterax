r"""pipekit-cycle protocol adapters.

`pipekit-cycle <https://github.com/jejjohnson/pipekit>`_ orchestrates
data-assimilation cycles through three runtime-checkable Protocols —
``ForwardModel``, ``ObservationOperator``, and ``AnalysisStep`` — and
algorithm libraries plug in *structurally*, by matching the protocol
signatures. The adapters here lift filterax components into those
shapes without filterax depending on pipekit:

* :class:`FilterAnalysisStep` — wraps any
  :class:`filterax.AbstractSequentialFilter` as a pipekit
  ``AnalysisStep`` (``__call__(forecast, obs, *, obs_op, obs_err_cov)``).
* :class:`DynamicsForwardModel` — wraps a
  :class:`filterax.AbstractDynamics` as a pipekit ``ForwardModel``
  (``step(state, dt)`` plus ``dt`` / ``state_signature`` attributes).
* :class:`LinearizableObsOperator` — wraps a
  :class:`filterax.AbstractObsOperator` (or plain callable) as a
  pipekit ``ObservationOperator``, deriving ``linearize`` from
  :func:`jax.jacfwd`.

pipekit's ``EnsembleDACycle`` represents the ensemble as a Python list
of member states; filterax stacks the ensemble along axis 0 of a single
array. :class:`FilterAnalysisStep` converts between the two at the
boundary, so the same filter object serves both worlds.

Randomness note: pipekit never passes a PRNG key to the analysis step,
so stochastic filters (e.g. ``StochasticEnKF``) draw from the key they
were constructed with. Prefer deterministic filters (ETKF, EnSRF) when
cycling through pipekit, or reconstruct the filter per cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from filterax._protocols import (
    AbstractDynamics,
    AbstractObsOperator,
    AbstractSequentialFilter,
)


def _as_noise_operator(obs_err_cov: Any) -> lx.AbstractLinearOperator:
    """Coerce pipekit's ``obs_err_cov`` payload to a lineax operator."""
    if obs_err_cov is None:
        raise ValueError(
            "filterax analysis steps need an observation-error covariance; "
            "set `obs_err_cov` on the DA state (e.g. DAState(obs_err_cov=R)) "
            "to a lineax operator or a dense (N_y, N_y) array."
        )
    if isinstance(obs_err_cov, lx.AbstractLinearOperator):
        return obs_err_cov
    matrix = jnp.asarray(obs_err_cov)
    return lx.MatrixLinearOperator(
        matrix, (lx.symmetric_tag, lx.positive_semidefinite_tag)
    )


class FilterAnalysisStep(eqx.Module, strict=True):
    r"""Expose a sequential filter as a pipekit ``AnalysisStep``.

    Satisfies ``pipekit_cycle.protocols.AnalysisStep`` structurally:
    ``__call__(forecast, obs, *, obs_op, obs_err_cov)``. The forecast
    ensemble may arrive either as pipekit's list of member states or as
    a stacked ``(Nₑ, Nₓ)`` array; the analysis ensemble is returned in
    the same container kind.

    Attributes:
        filter: Any :class:`filterax.AbstractSequentialFilter`
            (ETKF, EnSRF, LETKF, ...).
        analysis_kwargs: Extra keyword arguments forwarded to
            ``filter.analysis`` on every cycle — e.g. ``state_coords`` /
            ``obs_coords`` for :class:`filterax.filters.LETKF`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> from filterax.filters import ETKF
        >>> from filterax.pipekit import FilterAnalysisStep
        >>> step = FilterAnalysisStep(ETKF())
        >>> members = [jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])]
        >>> analysed = step(
        ...     members,
        ...     jnp.array([0.5, 0.5]),
        ...     obs_op=lambda x: x,
        ...     obs_err_cov=0.1 * jnp.eye(2),
        ... )
        >>> len(analysed), analysed[0].shape
        (2, (2,))
    """

    filter: AbstractSequentialFilter
    analysis_kwargs: dict[str, Any] = eqx.field(default_factory=dict)

    def __call__(
        self,
        forecast: Any,
        obs: Float[Array, " N_y"],
        *,
        obs_op: Callable[[Float[Array, " N_x"]], Float[Array, " N_y"]],
        obs_err_cov: Any,
    ) -> Any:
        as_members = isinstance(forecast, list | tuple)
        particles = jnp.stack(list(forecast)) if as_members else forecast
        result = self.filter.analysis(
            particles,
            obs,
            obs_op,
            _as_noise_operator(obs_err_cov),
            **self.analysis_kwargs,
        )
        return list(result.particles) if as_members else result.particles


class DynamicsForwardModel(eqx.Module, strict=True):
    r"""Expose autonomous dynamics as a pipekit ``ForwardModel``.

    Satisfies ``pipekit_cycle.protocols.ForwardModel`` structurally:
    ``step(state, dt)`` plus ``dt`` and ``state_signature`` attributes.
    pipekit does not thread absolute time into ``step``, so the wrapped
    dynamics are treated as autonomous and integrated over ``[0, dt]``.

    Attributes:
        dynamics: A :class:`filterax.AbstractDynamics` (or plain
            ``(state, t0, t1) -> state`` callable).
        dt: Default integration step advertised to pipekit.

    Examples:
        >>> import jax.numpy as jnp
        >>> from filterax.pipekit import DynamicsForwardModel
        >>> fwd = DynamicsForwardModel(lambda x, t0, t1: x + (t1 - t0), dt=0.5)
        >>> fwd.step(jnp.array([1.0]), fwd.dt)
        Array([1.5], dtype=float32)
    """

    dynamics: (
        AbstractDynamics
        | Callable[
            [Float[Array, " N_x"], Float[Array, ""], Float[Array, ""]],
            Float[Array, " N_x"],
        ]
    )
    dt: float = 1.0

    def step(self, state: Float[Array, " N_x"], dt: float) -> Float[Array, " N_x"]:
        """Advance one member state by ``dt`` (autonomous: ``t0 = 0``)."""
        return self.dynamics(state, jnp.asarray(0.0), jnp.asarray(dt))

    @property
    def state_signature(self) -> None:
        """No shape contract advertised; pipekit treats ``None`` as opt-out."""
        return None


class LinearizableObsOperator(eqx.Module, strict=True):
    r"""Expose an observation operator as a pipekit ``ObservationOperator``.

    Satisfies ``pipekit_cycle.protocols.ObservationOperator``
    structurally: ``__call__(state)`` plus ``linearize(state)``. The
    tangent-linear operator is derived with :func:`jax.jacfwd`, so any
    JAX-traceable ``H`` gets an exact Jacobian for free.

    Attributes:
        obs_op: A :class:`filterax.AbstractObsOperator` or plain
            ``state -> obs`` callable.

    Examples:
        >>> import jax.numpy as jnp
        >>> from filterax.pipekit import LinearizableObsOperator
        >>> H = LinearizableObsOperator(lambda x: x[:1] ** 2)
        >>> H(jnp.array([3.0, 4.0]))
        Array([9.], dtype=float32)
        >>> H.linearize(jnp.array([3.0, 4.0]))  # dH/dx = [2x, 0]
        Array([[6., 0.]], dtype=float32)
    """

    obs_op: AbstractObsOperator | Callable[[Float[Array, " N_x"]], Float[Array, " N_y"]]

    def __call__(self, state: Float[Array, " N_x"]) -> Float[Array, " N_y"]:
        return self.obs_op(state)

    def linearize(self, state: Float[Array, " N_x"]) -> Float[Array, "N_y N_x"]:
        """Jacobian ``∂H/∂x`` at ``state``, shape ``(N_y, N_x)``."""
        return jax.jacfwd(self.obs_op)(state)
