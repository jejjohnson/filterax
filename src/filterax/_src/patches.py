"""Spatial-decomposition primitives for patch-localized assimilation.

These primitives slice large gridded fields into overlapping patches,
route observations to their enclosing patches, and stitch per-patch
analyses back into a global field via distance-weighted blending. They
operate at the array level and remain decoupled from the L2 filter
models — Wave 4's ``LocalEnKF`` will consume them.

The decomposition itself is a host-side Python loop over patch origins
(``N_patches`` is a metadata quantity); the per-patch slicing returns
ordinary JAX arrays that can flow through ``jit`` / ``vmap`` downstream.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from filterax._src.localization import gaspari_cohn


class PatchMetadata(eqx.Module, strict=True):
    """Patch grid metadata needed to reconstruct a global field.

    ``origins`` stores the integer index of the lower corner of each patch
    along every spatial axis. ``patch_size`` and ``stride`` are length-``D``
    tuples; ``global_shape`` is the spatial shape of the source field
    (excluding any leading ensemble dimension).
    """

    origins: Int[Array, "N_patches D"]
    patch_size: tuple[int, ...] = eqx.field(static=True)
    stride: tuple[int, ...] = eqx.field(static=True)
    global_shape: tuple[int, ...] = eqx.field(static=True)


def _patch_origins_1d(extent: int, patch_size: int, stride: int) -> list[int]:
    """Per-axis origins: 0, stride, ..., max so that origin + patch_size <= extent."""
    if patch_size > extent:
        raise ValueError(
            f"patch_size={patch_size} exceeds extent={extent} along an axis."
        )
    last = extent - patch_size
    origins = list(range(0, last + 1, stride))
    if origins[-1] != last:
        origins.append(last)
    return origins


def create_patches(
    field: Float[Array, "N_e *spatial"],
    patch_size: tuple[int, ...],
    stride: tuple[int, ...],
) -> tuple[Float[Array, "N_patches N_e *patch_spatial"], PatchMetadata]:
    r"""Slice a leading-ensemble field into overlapping spatial patches.

    The first axis of ``field`` is treated as the ensemble axis; the
    remaining ``D`` axes are spatial and decomposed independently. Patches
    are taken at regular ``stride`` along each spatial axis, with a final
    flush-right patch added when ``stride`` does not divide the extent
    evenly so the global field is fully covered.

    Args:
        field: Array of shape ``(N_e, *spatial)``.
        patch_size: Length-``D`` patch size along each spatial axis.
        stride: Length-``D`` step between patch origins. ``stride < patch_size``
            produces overlapping patches.

    Returns:
        ``(patches, metadata)`` where ``patches`` has shape
        ``(N_patches, N_e, *patch_size)``.

    Raises:
        ValueError: if ``patch_size`` and ``stride`` disagree on dimensionality
            or any patch dimension exceeds the corresponding extent.
    """
    spatial = field.shape[1:]
    if len(patch_size) != len(spatial) or len(stride) != len(spatial):
        raise ValueError(
            "patch_size and stride must have one entry per spatial axis; "
            f"got patch_size={patch_size}, stride={stride}, "
            f"spatial={spatial}."
        )

    per_axis_origins = [
        _patch_origins_1d(extent, ps, st)
        for extent, ps, st in zip(spatial, patch_size, stride, strict=True)
    ]

    origins: list[tuple[int, ...]] = []
    patches: list[Float[Array, "..."]] = []
    for origin in _cartesian_product(per_axis_origins):
        origins.append(origin)
        slices = (
            slice(None),
            *(slice(o, o + ps) for o, ps in zip(origin, patch_size, strict=True)),
        )
        patches.append(field[slices])

    stacked = jnp.stack(patches, axis=0)
    metadata = PatchMetadata(
        origins=jnp.asarray(origins, dtype=jnp.int32),
        patch_size=tuple(patch_size),
        stride=tuple(stride),
        global_shape=tuple(spatial),
    )
    return stacked, metadata


def _cartesian_product(per_axis: list[list[int]]) -> list[tuple[int, ...]]:
    """Cartesian product of per-axis origin lists, lexicographic ordering."""
    if not per_axis:
        return [()]
    out: list[tuple[int, ...]] = [()]
    for axis in per_axis:
        out = [(*tail, val) for tail in out for val in axis]
    return out


def assign_obs_to_patches(
    obs_coords: Int[Array, "N_y D"],
    obs_values: Float[Array, " N_y"],
    metadata: PatchMetadata,
    buffer: int = 0,
) -> dict[int, tuple[Int[Array, "n_local D"], Float[Array, " n_local"]]]:
    r"""Bucket observations into the patches whose footprint contains them.

    For each patch ``p`` with origin ``o_p`` and footprint
    ``[o_p - buffer, o_p + patch_size + buffer)`` along every axis, return
    the observations falling inside. An observation may be assigned to
    multiple overlapping patches.

    Args:
        obs_coords: Integer grid indices of observations, shape ``(N_y, D)``.
        obs_values: Observation values, shape ``(N_y,)``.
        metadata: ``PatchMetadata`` produced by :func:`create_patches`.
        buffer: Extra grid points by which to extend each patch footprint
            (so observations near a patch boundary are picked up by both
            neighbours).

    Returns:
        Dict ``patch_id -> (local_coords, local_values)`` containing only
        patches with at least one observation.
    """
    origins_np = jnp.asarray(metadata.origins)
    patch_size = jnp.asarray(metadata.patch_size)
    obs_coords = jnp.asarray(obs_coords)

    assignments: dict[
        int, tuple[Int[Array, "n_local D"], Float[Array, " n_local"]]
    ] = {}
    for patch_id, origin in enumerate(origins_np):
        lower = origin - buffer
        upper = origin + patch_size + buffer
        in_box = jnp.all((obs_coords >= lower) & (obs_coords < upper), axis=1)
        idx = jnp.nonzero(in_box, size=int(in_box.size), fill_value=-1)[0]
        idx = idx[idx >= 0]
        if int(idx.shape[0]) == 0:
            continue
        assignments[patch_id] = (obs_coords[idx], obs_values[idx])
    return assignments


def blend_patches(
    patches: Float[Array, "N_patches N_e *patch_spatial"],
    metadata: PatchMetadata,
    taper_fn: Callable[
        [Float[Array, "..."], float], Float[Array, "..."]
    ] = gaspari_cohn,
    taper_radius: float | None = None,
) -> Float[Array, "N_e *spatial"]:
    r"""Stitch overlapping patches back into a global field.

    Each patch contributes its values weighted by a centred taper:

    .. math::

        x_{\text{global}}[i, k] = \frac{\sum_p w_p(k)\, x_p[i, k - o_p]}{\sum_p w_p(k)}

    where ``w_p(k) = taper_fn(d_p(k), r)`` and ``d_p(k)`` is the Euclidean
    distance from grid point ``k`` to the centre of patch ``p`` (restricted
    to grid points inside the patch). When ``taper_radius`` is ``None``,
    uniform weights are used (simple averaging in overlap regions).

    Args:
        patches: Per-patch fields ``(N_patches, N_e, *patch_size)``.
        metadata: ``PatchMetadata`` from :func:`create_patches`.
        taper_fn: Distance-to-weight function, signature ``(distances,
            radius) -> weights``. Defaults to :func:`gaspari_cohn`.
        taper_radius: Optional radius for the taper. When ``None`` all
            in-patch grid points receive weight ``1.0``.

    Returns:
        Reconstructed field of shape ``(N_e, *global_spatial)``.
    """
    n_ensemble = patches.shape[1]
    out_shape = (n_ensemble, *metadata.global_shape)
    accum = jnp.zeros(out_shape, dtype=patches.dtype)
    weight = jnp.zeros(metadata.global_shape, dtype=patches.dtype)

    centre = jnp.asarray(metadata.patch_size, dtype=patches.dtype) / 2.0 - 0.5

    for patch_id, origin in enumerate(metadata.origins):
        slices = (
            slice(None),
            *(
                slice(int(origin[d]), int(origin[d]) + metadata.patch_size[d])
                for d in range(len(metadata.patch_size))
            ),
        )

        # Per-cell weights for this patch.
        if taper_radius is None:
            w_local = jnp.ones(metadata.patch_size, dtype=patches.dtype)
        else:
            grids = jnp.meshgrid(
                *[jnp.arange(s, dtype=patches.dtype) for s in metadata.patch_size],
                indexing="ij",
            )
            sq = sum((g - centre[d]) ** 2 for d, g in enumerate(grids))
            d_local = jnp.sqrt(sq)
            w_local = taper_fn(d_local, taper_radius)

        accum = accum.at[slices].add(patches[patch_id] * w_local[None, ...])
        weight = weight.at[slices[1:]].add(w_local)

    safe = jnp.where(weight > 0, weight, 1.0)
    return accum / safe[None, ...]
