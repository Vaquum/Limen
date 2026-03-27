# `limen.experiment`

> Define experiments, run parameter search, and promote successful runs into reusable inference objects.

## Canonical docs

- [Experiment Manifest](../../docs/Experiment-Manifest.md)
- [Universal Experiment Loop](../../docs/Universal-Experiment-Loop.md)
- [Trainer](../../docs/Trainer.md)

## What this package owns

Owns the Manifest, the Universal Experiment Loop, advanced search infrastructure, checkpointing, and the retraining path that produces `Sensor` objects.
Does **not** own model architectures, indicators, features, or raw metric helpers.

## Key entry points

| Entry point | Use it when | Notes |
|-------------|-------------|-------|
| `Manifest` | You want a declarative experiment pipeline | Exported at the package root |
| `UniversalExperimentLoop` | You want to run an SFD across many permutations | Exported at the package root |
| `Trainer` | You want to rebuild winning rounds into reusable inference artifacts | Exported at the package root |
| `Sensor` | You want a portable inference object produced by `Trainer` | Exported at the package root |
| `ReconstructionError` | You need to handle failed manifest reconstruction during training | Exported at the package root |

## Adjacent modules

- `limen.sfd` supplies the experiment configuration that this package runs.
- `limen.data` supplies raw data when a manifest declares a data source.
- `limen.log` analyzes completed runs.
- `limen.cohort` builds decoder cohorts on top of finished experiment results.

## Quick orientation

```text
experiment/
├── experiment_core.py       # UniversalExperimentLoop
├── manifest_core.py         # Manifest and builder components
├── trainer.py               # Trainer, Sensor, ReconstructionError
├── checkpoint_manager.py    # Persist and restore advanced-run state
├── feedback_controller.py   # Mid-run adaptive callbacks
├── msq.py                   # Mutable Search Queue
├── param_domain.py          # Mutable parameter domain
├── reducer/
│   └── pruning_strategy.py  # Pruning interfaces and implementations
└── search_strategy.py       # SearchStrategy base class
```

## Things to know

- Manifest-driven SFDs require `prep_each_round=True`.
- The basic `run()` path uses legacy `ParamSpace`. The advanced path adds `SearchStrategy`, `ParamDomain`, `MSQ`, checkpoints, and feedback hooks.
- When `experiment_dir` is set, Limen writes all major artifacts under one directory, including checkpoint, audit, round-data, and results files.
- Results are appended incrementally during a run, so experiment files should be treated as durable artifacts rather than temporary output.

## Read next

- [Universal Experiment Loop](../../docs/Universal-Experiment-Loop.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md)
- [Trainer](../../docs/Trainer.md)
