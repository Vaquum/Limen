# `limen.experiment`

> Orchestrate multi-permutation hyperparameter search experiments, from parameter sampling through data prep to result logging.

## Responsibilities

Owns the main experiment loop (`UniversalExperimentLoop`), declarative experiment configuration (`Manifest`), parameter domain management, search strategies, checkpoint persistence, and feedback-driven adaptive search.
Does **not** own model implementations, feature engineering, or metrics computation — those are provided by the SFD and supporting modules.

## Key concepts

- **UniversalExperimentLoop (UEL)** – the central experiment runner; iterates `n_permutations` times, calling `prep` then `model` for each sampled parameter set, and writes results to a CSV and (optionally) SQLite.
- **Manifest** – declarative, fluent configuration object that wires together data source, features/indicators, target transforms, scaler, and model function; eliminates the need to write a custom `prep` function for most experiments.
- **SFD (Single File Decoder)** – a Python module (or object) that exposes `params()` and, for manifest-driven experiments, `manifest()`. SFDs live in `limen.sfd`.
- **ParamDomain** – mutable registry of all searchable parameter lists; strategies draw from it.
- **SearchStrategy** – abstract iterator that yields `dict[str, Any]` parameter combinations; concrete strategies (e.g. random, grid) implement `__next__`.
- **MSQ (Mutable Search Queue)** – a queue that strategies and feedback controllers can prepend to, enabling targeted re-evaluation.
- **CheckpointManager** – saves and restores `(MSQ, ParamDomain, round_number)` state to disk at configurable intervals, allowing long runs to be resumed after interruption.
- **FeedbackController** – triggers mid-run pruning or parameter re-weighting callbacks based on a temporary `Log` snapshot.
- **PruningStrategy** – decides which parameter combinations to drop based on live experiment results.

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `UniversalExperimentLoop` | `experiment_core.py` | Instantiate with `sfd=` (and optionally `data=`), then call `.run()` |
| `Manifest` | `manifest_core.py` | Built inside `sfd.manifest()` using the fluent builder API; consumed by UEL |

## Dependencies

- **Internal:** `limen.log` (creates a `Log` after the run), `limen.utils.param_space` (legacy `ParamSpace`), `limen.data` (auto-fetches data when manifest has a data source configured)
- **External:** `polars`, `tqdm`, `sqlite3` (stdlib)

## Quick orientation
```text
experiment/
├── experiment_core.py       # UniversalExperimentLoop — the main loop
├── manifest_core.py         # Manifest, DataSourceConfig, TargetBuilder, etc.
├── checkpoint_manager.py    # Save/load/validate checkpoint state
├── feedback_controller.py   # Mid-run adaptive search callbacks
├── msq.py                   # Mutable Search Queue
├── param_domain.py          # Mutable parameter domain + observer pattern
├── pruning_strategy.py      # Abstract and concrete pruning strategies
└── search_strategy.py       # Abstract SearchStrategy base class
```

## Gotchas / things to know

- `prep_each_round=True` is **required** for manifest-driven SFMs; UEL will raise if it is `False`.
- Manifest-driven experiments cannot override `prep` / `model` in `.run()` — only `params` can be overridden.
- If `LOOP_ENV=test`, UEL auto-fetches data via `manifest.fetch_test_data()` instead of the production data source.
- Results are appended to `{experiment_name}.csv` incrementally — the file is not overwritten between runs.
- Signal handlers (`SIGTERM`/`SIGINT`) are registered by `_register_shutdown_handler`; they set `_shutdown_requested=True` for a graceful single-run cutoff. A second signal raises `KeyboardInterrupt` immediately.
