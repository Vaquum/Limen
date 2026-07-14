# `limen.inference`

> Replay selected experiment rounds into validated models and run them bar-by-bar at inference time.

## Canonical docs

- [Trainer](../../docs/Trainer.md)

## What this package owns

Owns the reconstruction-and-inference path: replaying a selected permutation from a completed experiment, validating it against the original log, and wrapping it as a live `Sensor`.
Does **not** own experiment search execution (owned by `limen.experiment`) or the reference model implementations it reuses (owned by `limen.sfd.reference_architecture`).

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `Trainer` | Replay selected permutations from a completed YAML experiment | Re-runs `prepare_data()`/`run_model()` with the original split and compares metrics to the log |
| `Sensor` | Inference wrapper around a trained YAML model | Predicts one bar at a time for live use |
| `BarPrediction` | Result of a single-bar prediction | Carries `prediction`, `probability`, and a `reason` |
| `ReconstructionError` | Raised when metric validation deviates beyond tolerance | Signals a permutation could not be faithfully rebuilt |

## Adjacent modules

- `limen.experiment` produces the experiment logs and artifacts `Trainer` reconstructs. It does not re-export inference names; import them from `limen.inference` or top-level `limen`.
- `limen.sfd.reference_architecture.ReferenceModel` is the model type a `Sensor` wraps.
- `limen.yaml.compiler.CompiledSFD` supplies the compiled manifest used during reconstruction.

## Quick orientation

```text
inference/
├── trainer.py   # Trainer: reconstruct and validate a permutation
├── sensor.py    # Sensor + BarPrediction: live bar-by-bar prediction
└── errors.py    # ReconstructionError
```

## Things to know

- `Trainer` validates by re-running the manifest and comparing metrics to the original log; a deviation past tolerance raises `ReconstructionError` rather than returning a mismatched model.
- Metric comparison uses a float tolerance (with a wider tolerance for stochastic models), and a fixed set of bookkeeping columns (`id`, `execution_time`, …) is skipped during comparison.
- `BarPrediction.reason` explains a `None` prediction (`'warm-up'`, `'inside-training-window'`, `'null-features'`, `'sensor-error'`) — a null prediction is expected during warm-up, not an error.

## Read next

- [Trainer](../../docs/Trainer.md)
- [Single-File Decoder](../../docs/Single-File-Decoder.md)
