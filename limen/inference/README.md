# `limen.inference`

> Reconstruct compiled experiment rounds or wrap caller-owned Python fitted state for inference.

## Canonical docs

- [Trainer](../../docs/Trainer.md)

## In the research projects

The [Python project](../../docs/examples/python-research/README.md) promotes evaluated fitted objects without fitting them again. The [Compiled project](../../docs/examples/compiled-research/README.md) reconstructs selected recorded rounds and qualified folds through Trainer. The Sensor and Cohort references distinguish retained fitted state from reconstruction under a recorded definition.

## What this package owns

Owns the reconstruction-and-inference path: replaying a selected permutation from a completed experiment, validating it against the original log, and wrapping it as a live `Sensor`.
Also owns convenient `Sensor.from_model` and advanced `Sensor.from_python` construction without a Manifest. Defaults handle ordinary fitted prediction; optional functions retain custom preparation, prediction and output control. Caller declarations are distinct from compiled evidence.
Does **not** own experiment search execution (owned by `limen.experiment`) or the reference model implementations it reuses (owned by `limen.sfd.reference_architecture`).

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `Trainer` | Replay selected permutations from a completed YAML experiment | Re-runs `prepare_data()`/`run_model()` with the original split and compares metrics to the log |
| `Sensor` | Native inference over compiled or live Python fitted state | `from_model` wraps fitted models in one line; `from_python` accepts arbitrary state/history/callbacks |
| `BarPrediction` | Result of a single-bar prediction | Carries `prediction`, `probability`, and a `reason` |
| `ReconstructionError` | Raised when metric validation deviates beyond tolerance | Signals a permutation could not be faithfully rebuilt |

## Adjacent modules

- `limen.experiment` produces the experiment logs and artifacts `Trainer` reconstructs. It does not re-export inference names; import them from `limen.inference` or top-level `limen`.
- `limen.sfd.reference_architecture.ReferenceModel` is the compiled Sensor model contract; Python construction does not require inheritance.
- `limen.yaml.compiler.CompiledSFD` supplies the compiled manifest used during reconstruction.

## Quick orientation

| Module or family | Responsibility |
|---|---|
| `trainer.py` | Trainer: reconstruct and validate a permutation |
| `sensor.py` | Sensor entry points and BarPrediction re-export |
| `_python_sensor.py` | Python construction contract and shared result |
| `_sensor_from_model.py` | Fitted-model defaults and optional hooks |
| `errors.py` | ReconstructionError |

<!-- reference: package-inference-layout source=limen/inference/README.md region=layout:package role=layout -->
```text
inference/
├── README.md
├── __init__.py
├── _fold_metrics.py
├── _fold_sensor.py
├── _fold_trainer.py
├── _python_sensor.py
├── _sensor_from_model.py
├── _trainer_records.py
├── errors.py
├── sensor.py
└── trainer.py
```

## Things to know

- `Trainer` validates by re-running the manifest and comparing metrics to the original log; a deviation past tolerance raises `ReconstructionError` rather than returning a mismatched model.
- Metric comparison uses a float tolerance (with a wider tolerance for stochastic models), and a fixed set of bookkeeping columns (`id`, `execution_time`, …) is skipped during comparison.
- `BarPrediction.reason` explains a `None` prediction (`'warm-up'`, `'inside-training-window'`, `'null-features'`, `'sensor-error'`) — a null prediction is expected during warm-up, not an error.
- Python callback/adapter failures raise; invalid aligned values become `sensor-error`. Use `Cohort.from_python(selected_sensors)` for live composition; compiled `set_members` rejects Python-origin members. See the [Python contract](../../docs/Trainer.md#python-construction) and [recorded workflow](../../docs/Universal-Experiment-Loop.md#direct-python-run).

## Read next

- [Trainer](../../docs/Trainer.md)
- [Single-File Decoder](../../docs/Single-File-Decoder.md)

Recorded walk-forward promotion uses `Trainer(directory).train(ids, fold=k)`. It requires original source and versioned opt-in fold evidence, performs a validated refit, and guards the entire campaign history. See the canonical [Trainer contract](../../docs/Trainer.md#recorded-walk-forward-folds).
