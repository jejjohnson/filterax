---
status: draft
version: 0.1.0
---

# ekalmX — Research

Background research, prior art audits, and theoretical foundations that informed the design.

## Structure

```
research/
├── README.md                  # This file
├── research_enskf.md          # Audit of existing kalman_filter/ codebase (~5,000 lines)
└── ens_kalman_process.md      # EKP theory: math, numerical requirements, process zoo, BLR connection
```

## Reading Order

1. **[research_enskf.md](research_enskf.md)** — inventory of existing code being migrated (9 discrete filters, 9 continuous-time filters, patch decomposition, parameter estimation). Includes gap analysis and mapping to ekalmX architecture.

2. **[ens_kalman_process.md](ens_kalman_process.md)** — deep-dive on Ensemble Kalman Processes (EKI, EKS, UKI, ETKI, GNKI). Covers mathematical formulations, numerical requirements (ensemble size, scheduling, constraints), process zoo comparison tables, JAX API design, example applications, connection to the Bayesian Learning Rule, and gaussx integration.

## How These Relate to the Design Docs

| Research file | Informs | Key content |
|---|---|---|
| `research_enskf.md` | `api/components.md` (filters, smoothers), `boundaries.md` (gap analysis) | What exists today, what needs formalization |
| `ens_kalman_process.md` | `api/components.md` (processes, schedulers), `architecture.md` (gaussx integration), `boundaries.md` (BLR decision tree) | Math behind each process, when to use which |
