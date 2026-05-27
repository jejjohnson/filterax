"""Tests for the patch decomposition primitives."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import filterax as flx


def test_create_patches_1d_no_overlap():
    field = jnp.arange(20.0).reshape(1, 20)  # (N_e=1, N_x=20)
    patches, meta = flx.create_patches(field, patch_size=(5,), stride=(5,))
    assert patches.shape == (4, 1, 5)
    np.testing.assert_array_equal(
        np.asarray(meta.origins), np.asarray([[0], [5], [10], [15]])
    )
    np.testing.assert_array_equal(np.asarray(patches[0, 0]), np.arange(0, 5))
    np.testing.assert_array_equal(np.asarray(patches[-1, 0]), np.arange(15, 20))


def test_create_patches_1d_overlap_includes_flush_right():
    field = jnp.arange(12.0).reshape(1, 12)
    patches, meta = flx.create_patches(field, patch_size=(4,), stride=(3,))
    # origins: 0, 3, 6, 8 (last is flush right at 12 - 4 = 8).
    np.testing.assert_array_equal(
        np.asarray(meta.origins).ravel(), np.asarray([0, 3, 6, 8])
    )
    assert patches.shape == (4, 1, 4)


def test_create_patches_2d_shape(getkey):
    field = jr.normal(getkey(), (3, 8, 8))
    patches, meta = flx.create_patches(field, patch_size=(4, 4), stride=(4, 4))
    assert patches.shape == (4, 3, 4, 4)
    assert meta.global_shape == (8, 8)


def test_create_patches_rejects_oversized_patch():
    field = jnp.zeros((1, 4))
    try:
        flx.create_patches(field, patch_size=(5,), stride=(1,))
    except ValueError as exc:
        assert "exceeds extent" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_assign_obs_to_patches_in_bounds():
    field = jnp.zeros((1, 10, 10))
    _, meta = flx.create_patches(field, patch_size=(5, 5), stride=(5, 5))
    obs_coords = jnp.asarray([[1, 1], [6, 6], [3, 8], [8, 2]])
    obs_values = jnp.asarray([10.0, 20.0, 30.0, 40.0])
    assignments = flx.assign_obs_to_patches(obs_coords, obs_values, meta)
    # Should land in 4 distinct patches (one per corner).
    assert len(assignments) == 4


def test_assign_obs_to_patches_with_buffer_double_assigns():
    field = jnp.zeros((1, 10))
    _, meta = flx.create_patches(field, patch_size=(5,), stride=(5,))
    # Coordinate 4 sits in patch 0; with buffer=2 it also lands in patch 1.
    obs_coords = jnp.asarray([[4]])
    obs_values = jnp.asarray([1.0])
    no_buf = flx.assign_obs_to_patches(obs_coords, obs_values, meta, buffer=0)
    assert set(no_buf.keys()) == {0}
    with_buf = flx.assign_obs_to_patches(obs_coords, obs_values, meta, buffer=2)
    assert set(with_buf.keys()) == {0, 1}


def test_blend_patches_uniform_recovers_field():
    field = jnp.arange(24.0).reshape(2, 12)  # N_e=2
    patches, meta = flx.create_patches(field, patch_size=(4,), stride=(4,))
    blended = flx.blend_patches(patches, meta, taper_radius=None)
    np.testing.assert_allclose(np.asarray(blended), np.asarray(field), atol=1e-10)


def test_blend_patches_overlap_averages_via_uniform():
    field = jnp.arange(24.0).reshape(2, 12)
    patches, meta = flx.create_patches(field, patch_size=(6,), stride=(3,))
    blended = flx.blend_patches(patches, meta, taper_radius=None)
    # Uniform weight averaging in overlap regions still reproduces the
    # original since each cell is filled with copies of the same value.
    np.testing.assert_allclose(np.asarray(blended), np.asarray(field), atol=1e-10)
