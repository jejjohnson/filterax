---
status: draft
version: 0.1.0
---

# filterax × geostack — coordinate-aware ensembles

**Subject:** How to run filterax on ensembles whose particles carry CRS, named dimensions, and time coordinates (`coordax.Array` / `GeoTensor`).

**Decision anchors:** [D13 (carrier adapter)](../decisions.md#d13-coordinate-aware-carriers-via-adapter-not-core-type), [D14 (GeoLocalizer)](../decisions.md#d14-geospatial-localization-owned-by-filterax), [D16 (patcher)](../decisions.md#d16-patcher-based-localized-da-lives-in-filterax)

---

## 1  What geostack provides

The geostack ecosystem (`georeader`, `GeoCatalog`, `geotoolz`, `xrtoolz`) ships:

- `georeader.GeoTensor` — typed raster carrier with CRS + affine transform.
- `coordax.Array` — JAX-native coordinate-aware array (still maturing).
- `geotoolz.patch` (incubating) — spatial sampler + stitcher.

filterax does **not** ship any of these. The integration story is two seams:

1. **Carrier adapter** — turn a coordinate-aware ensemble into `Float[Array, "N_e N_x"]` for analysis, and back.
2. **Geographic localization** — consume coordinates as static metadata for `GeoLocalizer` (per D14).

## 2  The carrier adapter

Filters operate on flat ensemble arrays. Coordinate-aware carriers are handled outside the math layer by a `CarrierAdapter`:

```python
class CarrierAdapter(eqx.Module):
    """Round-trip a coordinate-aware carrier to/from FilterState.particles."""

    @abc.abstractmethod
    def flatten(self, carrier) -> tuple[Float[Array, "N_e N_x"], "CarrierMeta"]:
        ...

    @abc.abstractmethod
    def unflatten(self, particles, meta) -> Any:
        ...
```

filterax ships **one** reference implementation:

| Adapter | Carrier | When to use | Owner |
|---------|---------|-------------|-------|
| `DictAdapter` | `dict[str, Array]` of named state fields | Mixed-shape state (scalars + grids), no spatial coords needed | **filterax** |
| `CoordaxAdapter` | `coordax.Array` | Pure spatial state, coordinates needed for localization | downstream (geostack / plumax) |
| `GeoTensorAdapter` | `georeader.GeoTensor` | Raster particles with CRS + affine transform | downstream (geostack) |

filterax does **not** depend on `coordax`, `georeader`, or `pyproj` for the carrier path. Coordinate-aware adapters are recipe-level documentation here; the implementations live in the libraries that own those carrier types. §4 below shows the expected shape so downstream authors stay consistent with filterax's interface.

## 3  DictAdapter recipe

For a plumax-style state that mixes scalar parameters and gridded fields:

```python
import filterax
import jax.numpy as jnp

adapter = filterax.integrations.DictAdapter(
    schema={
        "Q":      (4,),        # 4 source rates
        "x0":     (4, 2),      # 4 source positions
        "c_bg":   (),          # background scalar
        "b_inst": (3,),        # 3 instrument biases
    }
)

ensemble = {
    "Q":      jnp.zeros((N_e, 4)),
    "x0":     jnp.zeros((N_e, 4, 2)),
    "c_bg":   jnp.zeros((N_e,)),
    "b_inst": jnp.zeros((N_e, 3)),
}

particles, meta = adapter.flatten(ensemble)   # (N_e, 4 + 8 + 1 + 3) = (N_e, 16)

# ... analysis on flat particles ...

new_ensemble = adapter.unflatten(updated_particles, meta)
```

The schema is **static**, captured in `meta` as a tuple of `(name, shape, slice)` records. JIT picks up the slicing as static structure; the flatten/unflatten round-trip is essentially free at runtime.

## 4  CoordaxAdapter recipe (downstream-owned)

Shape that a `coordax.Array` adapter should follow when authored in geostack / plumax / user code. **filterax does not ship this class** — it is documented here so downstream authors stay consistent with the `CarrierAdapter` interface in §2:

```python
import coordax
import jax.numpy as jnp
import filterax

class CoordaxAdapter(filterax.CarrierAdapter):
    """Round-trip a coordax.Array ensemble carrier (downstream / user-owned)."""

    ensemble_dim: str = "ensemble"

    def flatten(self, carrier):
        # ... extract data, dims, coords; reshape to (N_e, N_x) ...
        ...

    def unflatten(self, particles, meta):
        # ... rebuild coordax.Array from data + saved dims/coords ...
        ...

state = coordax.Array(
    data=jnp.zeros((N_e, n_lat, n_lon)),
    dims=("ensemble", "lat", "lon"),
    coords={"lat": lat_array, "lon": lon_array},
)

adapter = CoordaxAdapter(ensemble_dim="ensemble")
particles, meta = adapter.flatten(state)
# particles: (N_e, n_lat * n_lon)
# meta: CarrierMeta(dims=("lat", "lon"), shape=(n_lat, n_lon), coords={...})

# ... analysis ...

state_a = adapter.unflatten(updated_particles, meta)
# state_a is a coordax.Array with the same dims and coords as input
```

The adapter preserves dim order and coordinate arrays; only the underlying data buffer is replaced. Authoring this in geostack or plumax keeps `coordax` (and any other carrier-side dependency) out of filterax's install.

## 5  Geographic localization

`GeoLocalizer` consumes coordinates as static metadata (see `features/localization_inflation.md` §A.2.1 and D14):

```python
import filterax
import numpy as np

# coords: (N_x, 2) lon/lat array of state grid points
coords = np.column_stack([lon_array.ravel(), lat_array.ravel()])

frame = filterax.LocalFrame.from_crs(
    src_crs="EPSG:4326",
    dst_crs="EPSG:32633",      # UTM zone 33N — projection appropriate for AOI
    origin=(lon0, lat0),
)

localizer = filterax.GeoLocalizer(
    coords=coords,
    frame=frame,
    radius_km=50.0,
    taper="gaspari_cohn",
)

filter_ = filterax.LETKF(localizer=localizer, ...)
```

`LocalFrame.from_crs` is the only step that calls `pyproj`; it runs once at construction time. The `GeoLocalizer.__call__` path is pure JAX.

## 6  Patcher integration

filterax ships an in-house `RegularGridPatcher` in Wave 2 and an `AbstractPatcher` extension point in Wave 4 (D16). When `geotoolz.patch` stabilizes, geostack users can plug it in directly:

```python
from geotoolz.patch import GridSampler, Stitcher       # post-stabilization

class GeoPatcher(filterax.AbstractPatcher):
    sampler: GridSampler = eqx.field(static=True)
    stitcher: Stitcher = eqx.field(static=True)

    def patches(self, state):
        return self.sampler(state)

    def stitch(self, patches, shape):
        return self.stitcher(patches, shape)

local_filter = filterax.LocalEnKF(
    base_filter=filterax.LETKF(...),
    patcher=GeoPatcher(sampler=..., stitcher=...),
)
```

Until then, `LocalEnKF` consumes the in-house patcher; the user-facing API is identical.

## 7  Things filterax deliberately does *not* ship

- **No coordax / GeoTensor / pyproj** in the required dependency set. All three are soft-optional.
- **No xarray adapter.** xarray orchestration belongs to `xr_assimilate` (see `boundaries.md`).
- **No CRS reprojection.** Pre-project coordinates before constructing the localizer; reprojection inside the analysis path is not a filterax concern.
- **No catalog / I/O.** `GeoCatalog`, `georeader`, and friends are upstream of filterax.

## 8  Migration outlook

The carrier adapter pattern is intentionally minimal so filterax can absorb whatever coordinate library wins out (`coordax`, `xarray`, `GeoTensor`). The interface is short enough that an adapter for a new carrier is ~30 lines of user code.

If a future ecosystem decision settles on one carrier library, filterax can promote that adapter to the core install — without changing the math layer.
