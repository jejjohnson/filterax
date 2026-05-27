---
status: draft
version: 0.1.0
---

# Integrations

Recipes for plugging filterax into the broader ecosystem. Each doc here is
**recipe-shaped** — it shows how to use filterax with another library, not how
to extend filterax internally.

| Doc | Covers | Decision anchor |
|-----|--------|-----------------|
| [pipekit.md](pipekit.md) | Structural Protocol satisfaction with `pipekit-cycle`; using filterax filters inside pipekit `Sequential` / `Graph` / `Cycle` pipelines. | D11 |
| [plumax.md](plumax.md) | Multi-instrument methane attribution (Tier IV): TROPOMI + EMIT + GHGSat fusion with `JointObsOperator`, `SequentialAssimilation`, fixed-lag smoother. | D8, D12 |
| [geostack.md](geostack.md) | `CarrierAdapter` for coordinate-aware particles (`coordax.Array`, `GeoTensor`); `LocalFrame` construction; `GeoLocalizer` use. | D13, D14 |

## What lives here vs. elsewhere

- **`integrations/`** (this directory): how filterax composes with named external libraries (pipekit, plumax, geostack). Recipe form.
- **`examples/integration.md`**: cross-cutting composition patterns (optax, somax, gaussx) that are not tied to a single downstream library.
- **`boundaries.md`**: the ownership table that says which library owns which concern. Read this first if you want to know whether a feature belongs in filterax.

## Reading order for a downstream library author

1. [boundaries.md](../boundaries.md) — confirm filterax is the right place to plug in
2. [pipekit.md](pipekit.md) — if your pipeline is a pipekit `Sequential` / `Graph`
3. [geostack.md](geostack.md) — if your particles are coordinate-aware
4. [plumax.md](plumax.md) — if you are fusing multiple observation streams
