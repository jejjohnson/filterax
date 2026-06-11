# Protocols & Types

filterax components talk to each other through a small set of abstract
interfaces (all `equinox.Module` subclasses, so implementations are immutable
pytrees that survive `jit` / `grad` / `vmap`) and a matching set of typed
containers. Implement the `Abstract*` classes to plug your own dynamics,
observation operators, localizers, or inflators into the Layer-2 loops; the
containers are what those loops accept and return.

## Component protocols

The pieces a data-assimilation problem is assembled from: dynamics propagate
a state between two times, observation operators map state space to
observation space, and noise models expose a covariance operator plus a
sampler.

::: filterax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [AbstractDynamics, AbstractObsOperator, AbstractNoise]

## Regularisation protocols

Localizers and inflators are the two standard fixes for small-ensemble rank
deficiency; schedulers control the artificial-time step of the ensemble
Kalman processes. Concrete implementations live on the
[Localization](localization.md), [Inflation](inflation.md), and
[Schedulers](schedulers.md) pages.

::: filterax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [AbstractLocalizer, AbstractInflator, AbstractScheduler]

## Algorithm protocols

The two top-level algorithm families: sequential filters expose a one-shot
`analysis` step (see [Filters](filters.md)), and ensemble Kalman processes
expose an `init` / `update` iteration (see [Processes](processes.md)).

::: filterax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [AbstractSequentialFilter, AbstractProcess]

## State & result containers

What flows through and out of the algorithms: per-cycle filter state, the
one-shot `AnalysisResult`, the full-trajectory `AssimilationResult` (and its
latent-space variant), the smoothers' `SmoothingResult`, and the process
iteration states.

::: filterax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [FilterState, AnalysisResult, AssimilationResult, LatentAssimilationResult, SmoothingResult, ProcessState, UKIState]

## Configuration

::: filterax
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [FilterConfig, ProcessConfig]
