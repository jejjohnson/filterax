"""Protocol smoke tests.

The abstract classes must refuse instantiation until concrete subclasses
override the required methods.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
import pytest
from jaxtyping import Array, Float

import filterax as flx


def test_abstract_dynamics_cannot_instantiate():
    with pytest.raises(TypeError):
        flx.AbstractDynamics()


def test_abstract_obs_operator_cannot_instantiate():
    with pytest.raises(TypeError):
        flx.AbstractObsOperator()


def test_abstract_sequential_filter_cannot_instantiate():
    with pytest.raises(TypeError):
        flx.AbstractSequentialFilter()


def test_abstract_process_cannot_instantiate():
    with pytest.raises(TypeError):
        flx.AbstractProcess()


def test_abstract_scheduler_cannot_instantiate():
    with pytest.raises(TypeError):
        flx.AbstractScheduler()


class _IdentityDynamics(flx.AbstractDynamics):
    def __call__(
        self,
        state: Float[Array, " N_x"],
        t0: Float[Array, ""],
        t1: Float[Array, ""],
    ) -> Float[Array, " N_x"]:
        return state


class _LinearObs(flx.AbstractObsOperator):
    H: Float[Array, "N_y N_x"]

    def __call__(
        self,
        state: Float[Array, " N_x"],
    ) -> Float[Array, " N_y"]:
        return self.H @ state


def test_concrete_dynamics_instantiates_and_runs():
    dyn = _IdentityDynamics()
    x = jnp.arange(4.0)
    out = dyn(x, jnp.asarray(0.0), jnp.asarray(1.0))
    assert jnp.allclose(out, x)


def test_concrete_obs_operator_vmaps_over_ensemble():
    obs_op = _LinearObs(H=jnp.eye(3, 5))
    particles = jnp.arange(20.0).reshape(4, 5)
    out = eqx.filter_vmap(obs_op)(particles)
    assert out.shape == (4, 3)


def test_protocols_are_equinox_modules():
    # Spot-check that the abstract bases inherit from eqx.Module so that any
    # concrete subclass is automatically a PyTree.
    for cls in (
        flx.AbstractDynamics,
        flx.AbstractObsOperator,
        flx.AbstractNoise,
        flx.AbstractLocalizer,
        flx.AbstractInflator,
        flx.AbstractScheduler,
        flx.AbstractSequentialFilter,
        flx.AbstractProcess,
    ):
        assert issubclass(cls, eqx.Module)


def test_concrete_noise_model_instantiates():
    class DiagNoise(flx.AbstractNoise):
        variance: Float[Array, " N_y"]

        def covariance(self) -> lx.AbstractLinearOperator:
            return lx.DiagonalLinearOperator(self.variance)

        def sample(self, key, shape):
            import jax.random as jr

            return jr.normal(key, shape) * jnp.sqrt(self.variance)

    noise = DiagNoise(variance=jnp.asarray([0.1, 0.2, 0.3]))
    cov = noise.covariance()
    assert cov.as_matrix().shape == (3, 3)
