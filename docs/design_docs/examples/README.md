---
status: draft
version: 0.1.0
---

# ekalmX — Examples

Usage patterns and integration snippets organized by API layer.
Each file is self-contained and shows how to use that layer of the library.

## Structure

```
examples/
├── README.md              # This file — overview and index
├── primitives.md          # Layer 0 — ensemble statistics, gain, localization, inflation, patches
├── components.md          # Layer 1 — building filters/processes, custom protocols, smoothers
├── models.md              # Layer 2 — LETKF/EKI/EKS with minimal boilerplate
└── integration.md         # Layer 3 — optax, somax, gaussx, geo_toolz, xr_assimilate, differentiable training
```

## Reading Order

Work bottom-up through the layers:

1. **[primitives.md](primitives.md)** — L0: how individual functions work
2. **[components.md](components.md)** — L1: how building blocks snap together
3. **[models.md](models.md)** — L2: how to use the high-level API
4. **[integration.md](integration.md)** — L3: how to compose with external packages
