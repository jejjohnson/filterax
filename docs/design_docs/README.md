---
status: draft
version: 0.1.0
---

# ekalmX Design Doc

**Differentiable ensemble Kalman methods for filtering, inversion, and joint estimation in JAX/Equinox.**

## Structure

```
filterX/
├── README.md              # This file
├── vision.md              # Motivation, user stories, design principles, identity
├── architecture.md        # Three-layer stack, protocols, data types, package layout, dependencies
├── boundaries.md          # Ownership, ecosystem, scope, testing strategy, roadmap
├── decisions.md           # 10 ADRs (D1–D10)
├── api/
│   ├── README.md          # Surface inventory, notation, performance notes
│   ├── primitives.md      # Layer 0 — ensemble statistics, gain, localization, inflation, patches
│   ├── components.md      # Layer 1 — protocols, filters, processes, smoothers, schedulers
│   └── models.md          # Layer 2 — LETKF/ETKF/EnSRF, EKI/EKS/UKI, optax transforms
├── features/
│   ├── filters.md         # Sequential EnKF variants (ETKF, LETKF, EnSRF, ESTKF, etc.)
│   ├── processes.md       # EKP algorithms (EKI, EKS, UKI, GNKI, SparseEKI, TEKI)
│   ├── smoothers.md       # Ensemble smoothers (EnKS, RTS, fixed-lag, IES)
│   ├── localization_inflation.md  # Localization tapers + inflation strategies
│   ├── diagnostics.md     # DA evaluation metrics (rank histogram, Desroziers, CRPS, etc.)
│   ├── differentiable_da.md      # Deep dive: backprop through filters
│   └── optax_ekp.md       # Deep dive: EKP as optax GradientTransformations
├── examples/
│   ├── README.md          # Index and reading order
│   ├── primitives.md      # Layer 0 usage patterns
│   ├── components.md      # Layer 1 composition patterns
│   ├── models.md          # Layer 2 high-level API patterns
│   └── integration.md     # Layer 3 cross-library patterns (optax, somax, gaussx, geo_toolz)
├── research/
│   ├── README.md          # Index for research docs
│   ├── research_enskf.md  # Audit of existing kalman_filter/ code (~5,000 lines)
│   └── ens_kalman_process.md  # EKP theory: math, process zoo, BLR connection
└── decisions.md           # 10 ADRs (D1–D10)
```

## Reading Order

1. **[vision.md](vision.md)** — understand the why
2. **[architecture.md](architecture.md)** — understand the three-layer stack
3. **[boundaries.md](boundaries.md)** — understand the scope
4. **[api/README.md](api/README.md)** — scan the surface
5. **[api/primitives.md](api/primitives.md)** → **[components.md](api/components.md)** → **[models.md](api/models.md)** — drill into detail
6. **[features/filters.md](features/filters.md)** → **[processes.md](features/processes.md)** → **[smoothers.md](features/smoothers.md)** — algorithm deep dives
7. **[features/localization_inflation.md](features/localization_inflation.md)** → **[diagnostics.md](features/diagnostics.md)** — tuning and evaluation
8. **[features/differentiable_da.md](features/differentiable_da.md)** → **[optax_ekp.md](features/optax_ekp.md)** — unique capabilities
9. **[examples/primitives.md](examples/primitives.md)** → **[components.md](examples/components.md)** → **[models.md](examples/models.md)** → **[integration.md](examples/integration.md)** — see it in action
10. **[decisions.md](decisions.md)** — understand the tradeoffs (D1–D10)
11. **[research/](research/)** — background theory and code audits
