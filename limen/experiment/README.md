# `limen.experiment`

> Define manifests and run basic or adaptive parameter search.

## Canonical docs

- [Experiment Manifest](../../docs/Experiment-Manifest.md)
- [Universal Experiment Loop](../../docs/Universal-Experiment-Loop.md)
- [Advanced Search](../../docs/Advanced-Search.md)
- [Reducers And Feedback](../../docs/Reducers-And-Feedback.md)

## What this package owns

Owns `Manifest`, `MLManifest`, `RuleBasedManifest`, the Universal Experiment Loop, parameter-search strategies, mutable search state, reducers, feedback, and checkpointing.
Does **not** own inference reconstruction (`limen.inference`), model architectures, indicators, features, or metric helpers.

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `Manifest`, `MLManifest`, `RuleBasedManifest`, `CalibrationConfig` | Declarative experiment pipelines and calibration configuration | Exported at the package root |
| `UniversalExperimentLoop` | Run an SFD across parameter permutations | Exported at the package root |
| `GridStrategy`, `RandomStrategy`, `STRATEGY_REGISTRY` | Built-in advanced-search strategy surface | Exported at the package root |

## Adjacent modules

- `limen.sfd` supplies the experiment configuration that this package runs.
- `limen.data` supplies raw data when a manifest declares a data source.
- `limen.log` analyzes completed runs.
- `limen.cohort` builds decoder cohorts on top of finished experiment results.
- `limen.inference` owns `Trainer`, `Sensor`, and `ReconstructionError`; the top-level `limen` package lazily re-exports those names.

## Quick orientation

```text
experiment/
├── experiment_core.py       # UniversalExperimentLoop
├── manifest_core.py         # Manifest and builder components
├── checkpoint_manager.py    # Persist and restore advanced-run state
├── feedback_controller.py   # Mid-run adaptive callbacks
├── msq.py                   # Mutable Search Queue
├── param_domain.py          # Mutable parameter domain
├── reducer/
│   └── pruning_strategy.py  # Pruning interfaces and implementations
└── param_search/            # SearchStrategy, GridStrategy, RandomStrategy
```

## Things to know

- Manifest-driven SFDs require `prep_each_round=True`.
- The basic `run()` path uses legacy `ParamSpace`. The advanced path adds `SearchStrategy`, `ParamDomain`, `MSQ`, checkpoints, and feedback hooks.
- When `experiment_dir` is set, Limen writes all major artifacts under one directory, including checkpoint, audit, round-data, and results files.
- Results are appended incrementally during a run, so experiment files should be treated as durable artifacts rather than temporary output.

## Read next

- [Universal Experiment Loop](../../docs/Universal-Experiment-Loop.md)
- [Advanced Search](../../docs/Advanced-Search.md)
- [Reducers And Feedback](../../docs/Reducers-And-Feedback.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md)
