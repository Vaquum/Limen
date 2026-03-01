# `limen.experiment`

> Run repeated model-parameter permutations, record results, and support checkpointing and adaptive search.

## Responsibilities

Owns the core experiment loop (`UniversalExperimentLoop`), the declarative pipeline specification (`Manifest`), and the infrastructure for search strategy, domain mutation, and checkpointing.

Does **not** own metrics computation, backtesting, or logging analysis — those live in `limen.metrics` and `limen.log`.

## Key concepts

- **UniversalExperimentLoop (UEL)** – drives `n_permutations` rounds; each round samples params, calls `prep` + `model`, accumulates results into a CSV and a Polars DataFrame, and attaches a `Log` object when done
- **Manifest** – fluent builder for declarative SFD pipelines; chains data source, bar formation, feature transforms, target transforms, scaler, and model function; drives `prepare_data()` and `run_model()` for UEL
- **ParamDomain** – mutable parameter space (dict of name → list of values); notifies observers when values are added/removed
- **SearchStrategy** – abstract iterator over `ParamDomain`; concrete strategies include random and grid search
- **MSQ (Mutable Search Queue)** – wraps a `SearchStrategy` and provides live intervention API (`remove_ge`, `keep_between`, `inject`, `trim`, etc.)
- **CheckpointManager** – saves/loads/validates experiment state to `checkpoint.json` using write-then-rename for atomicity
- **FeedbackController** – coordinates mid-run callbacks that inspect the live `Log` and mutate `MSQ`

## Entry points

| What | Where | When you'd call it |
|------|-------|--------------------|
| `UniversalExperimentLoop` | `experiment_core.py` | Instantiate with `sfd=`, then call `.run()` to execute the experiment |
| `Manifest` | `manifest_core.py` | Build pipeline declaratively inside an SFD's `manifest()` method |
| `ParamDomain` | `param_domain.py` | Created internally by UEL; accessed via `MSQ` for feedback-driven pruning |
| `MSQ` | `msq.py` | Returned by `FeedbackController`; call its mutation methods inside pruning callbacks |
| `CheckpointManager` | `checkpoint_manager.py` | Used by UEL internally; also usable standalone for resume logic |

## Dependencies

- **Internal:** `limen.log.Log` (attached after run), `limen.utils.ParamSpace`, `limen.data` (via Manifest data source)
- **External:** `polars`, `tqdm`, `sqlite3` (optional persistence)

## Quick orientation

```text
experiment/
├── experiment_core.py      # UniversalExperimentLoop — the main run loop
├── manifest_core.py        # Manifest + builder helpers (Manifest, TargetBuilder, etc.)
├── param_domain.py         # ParamDomain — mutable param space with observer pattern
├── search_strategy.py      # SearchStrategy ABC
├── msq.py                  # MSQ — mutable search queue with intervention API
├── checkpoint_manager.py   # CheckpointManager — atomic save/load/validate
├── feedback_controller.py  # FeedbackController — mid-run callback orchestration
└── pruning_strategy.py     # Built-in pruning strategy implementations
```

## Gotchas / things to know

- For manifest-driven SFDs, `prep_each_round=True` is **required**; passing `prep` or `model` overrides is disallowed
- Results are streamed to `<experiment_name>.csv` each round before the in-memory DataFrame is updated; the file survives crashes
- `save_to_sqlite=True` also writes to `/opt/experiments/experiments.sqlite`
- Checkpoint content hashes include the full parameter space; changing any param value between runs will cause `validate()` to raise and prevent accidental resume on a different config
- `LOOP_ENV=test` env var causes UEL to call `manifest.fetch_test_data()` instead of the production data source
