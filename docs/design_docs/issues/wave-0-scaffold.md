# [Wave 0] Scaffold

## Shared Context
This grouped draft is based on `README.md`, `vision.md`, `architecture.md`,
`boundaries.md`, and `decisions.md` from the `filterX` design-doc set. The repo
you want to build, though, is `filterax`.

The current `filterax` repository is still a package template:

- `pyproject.toml` still says `mypackage`
- `src/mypackage/` still exists
- `README.md` still points at `pypackage_template`
- there are no real runtime dependencies yet

The design docs expect a real JAX/Equinox data-assimilation library with three
progressive layers:

- Layer 0: pure ensemble/Kalman primitives
- Layer 1: protocols and configurable components
- Layer 2: ready-to-run models and optax wrappers

Wave 0 exists only to make the repo identity match that future.

## Canonical Package Layout

```text
src/filterax/
  __init__.py
  _src/
    _protocols.py
    _types.py
    statistics.py
    gain.py
    likelihood.py
    localization.py
    inflation.py
    perturbations.py
    patches.py
  filters/
  processes/
  smoothers/
  optax/
  utils/
```

## Runtime Boundary
The design docs are explicit about ownership:

- required runtime: `jax`, `equinox`, `gaussx`, `lineax`, `optax`
- expected near-core typing/utilities: `jaxtyping`
- optional ecosystem integrations: `diffrax`, `somax`, `geo_toolz`, `xr_assimilate`

Wave 0 should preserve that dependency boundary instead of flattening
everything into one base requirement list.

# [Wave 0] Scaffold and Package Identity
Draft ID: `FLX-01`
## Goal
Turn the template repo into a real `filterax` package with stable metadata,
package layout, and runtime dependency boundaries for the later waves.

## Why This Wave Exists
Until the template identity is gone, every later issue would mix real work with
rename cleanup, import churn, and broken docs metadata.

## Wave / Milestone
- Wave: `wave:0`
- Milestone: `v0.0-scaffold`

## Canonical Epics
- [ ] FLX-02 [Epic] 0.A Package Identity and Runtime Bootstrap

## Sequential Dependencies
- Single-theme wave: FLX-02 is the entire wave.
- `FLX-03 -> FLX-04 -> FLX-05`.
- The dedicated Core wave should not start until this scaffold wave closes.

## Definition of Done
- `import filterax` succeeds.
- The source tree, docs, metadata, and repo URLs no longer say `mypackage` or
  `pypackage_template`.
- The repo layout makes the Layer 0 / Layer 1 / Layer 2 architecture obvious.

## Relationships
- Blocks the dedicated Core wave and every later filter/process/smoother wave.

---

# [Epic] 0.A Package Identity and Runtime Bootstrap
Draft ID: `FLX-02`
## Theme
Replace the template repo identity with real `filterax` packaging, docs
positioning, and runtime dependency boundaries.

## Parent Wave
- Wave epic: `FLX-01`
- Wave label: `wave:0`
- Milestone: `v0.0-scaffold`

## Parallelism
- Can run in parallel with: none
- Blocked by (inside this wave): none
- Must complete before: FLX-06

## Definition of Done
- The repo is recognizably `filterax`.
- Core runtime dependencies are staged correctly.
- The public package namespace exists even if most internals are still stubs.

## Relationships
- Parent wave: FLX-01.
- Blocks FLX-06.

---

# build(package): rename mypackage scaffold to filterax and create the package layout
Draft ID: `FLX-03`
## Problem / Request
Rename the template scaffold to `filterax`, update metadata, and create the
package layout expected by the architecture doc.

## User Story
As a later implementer, I want the package to already import as `filterax` with
obvious subpackages for filters, processes, smoothers, optax, and utilities.

## Motivation
Every later issue assumes the repo already looks like a data-assimilation
library, not a freshly cloned template.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/architecture.md`, `design_docs/filterX/vision.md`
- Existing code to replace: `filterax/src/mypackage`, `filterax/pyproject.toml`, `filterax/README.md`

## Implementation Steps
- [ ] Rename `src/mypackage` to `src/filterax`.
- [ ] Create importable namespace packages or modules for `_src`, `filters`,
      `processes`, `smoothers`, `optax`, and `utils`.
- [ ] Update wheel/build/test/coverage paths to the new package name.
- [ ] Remove template-facing README/docs references.

## Definition of Done
- `import filterax` and the top-level namespace packages succeed.
- The package layout matches the design direction closely enough that later
  issues are implementation-only.

## Relationships
- Parent epic: FLX-02.
- Blocks FLX-04 and FLX-06.

---

# build(deps): add required runtime deps and staged optional integrations
Draft ID: `FLX-04`
## Problem / Request
Add the runtime dependency surface implied by the design docs without pulling in
all ecosystem integrations as hard requirements.

## User Story
As an implementer, I want the repo dependencies to match the architectural
boundary so later waves can rely on JAX/Equinox/Kalman infrastructure being
present.

## Motivation
The dependency decision is part of the design: `gaussx` and `optax` are core,
while `somax`, `geo_toolz`, and `xr_assimilate` are integrations.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/boundaries.md`, `design_docs/filterX/decisions.md`
- Existing code to replace: `filterax/pyproject.toml`

## Implementation Steps
- [ ] Add required runtime deps: `jax`, `equinox`, `gaussx`, `lineax`, `optax`,
      `jaxtyping`.
- [ ] Keep integration libraries such as `diffrax`, `somax`, `geo_toolz`, and
      `xr_assimilate` optional or docs-only.
- [ ] Reflect the package rename in project URLs and metadata.
- [ ] Preserve a clean split between runtime, docs/examples, and dev tooling.

## Definition of Done
- The project environment reflects the documented runtime boundary.
- Optional integrations are not forced into the base install.

## Relationships
- Parent epic: FLX-02.
- Blocked by FLX-03.
- Blocks FLX-06 and all implementation waves.

---

# docs: replace template README and docs landing with filterax positioning
Draft ID: `FLX-05`
## Problem / Request
Replace the template docs and README with the actual `filterax` identity and
layered architecture story.

## User Story
As a later reader or implementing agent, I want the repo landing docs to
describe `filterax` as a differentiable ensemble-Kalman library instead of a
generic Python template.

## Motivation
The README is part of the issue pack’s implementation context. It should state
the package identity, layering, and ecosystem boundaries from the start.

## References & Existing Code
- Design doc / spec: `design_docs/filterX/README.md`, `design_docs/filterX/vision.md`
- Existing code to replace: `filterax/README.md`, `filterax/docs/index.md`, `filterax/mkdocs.yml`

## Implementation Steps
- [ ] Rewrite the README around the one-liner, user stories, and three-layer
      stack from the design docs.
- [ ] Update the docs landing page and nav to reflect the upcoming wave split.
- [ ] Remove template badges, URLs, and package names.

## Definition of Done
- The landing docs explain `filterax` clearly enough that later issue bodies do
  not have to compensate for template confusion.

## Relationships
- Parent epic: FLX-02.
- Blocked by FLX-04.

