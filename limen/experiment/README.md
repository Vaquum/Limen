# `limen.experiment`

> Define manifests and run basic or adaptive parameter search.

## Canonical docs

- [Experiment Manifest](../../docs/Experiment-Manifest.md)
- [Universal Experiment Loop](../../docs/Universal-Experiment-Loop.md)
- [Advanced Search](../../docs/Advanced-Search.md)
- [Reducers And Feedback](../../docs/Reducers-And-Feedback.md)

## In the research projects

The [Python project](../../docs/examples/python-research/README.md) supplies its own SFD functions and bounded parameter domain. The [Compiled project](../../docs/examples/compiled-research/README.md) supplies a Manifest-backed SFD and saved definition. Both execute through UEL/MSQ; Compiled Research supplies YAML to the CLI; direct Manifest construction is optional component integration.

## What this package owns

Owns `Manifest`, `MLManifest`, `RuleBasedManifest`, the Universal Experiment Loop, parameter-search strategies, mutable search state, reducers, feedback, and checkpointing.
Does **not** own inference reconstruction (`limen.inference`), model architectures, indicators, features, or metric helpers.

## Key entry points

| Entry point | Use case | Notes |
|-------------|-------------|-------|
| `Manifest`, `MLManifest`, `RuleBasedManifest`, `CalibrationConfig` | Declarative experiment pipelines and calibration configuration | Exported at the package root |
| `UniversalExperimentLoop` | Run an SFD across parameter permutations | Exported at the package root |
| `GridStrategy`, `RandomStrategy`, `STRATEGY_REGISTRY` | Built-in advanced-search strategy surface | Exported at the package root |
| `ReuseContract`, `declare_reuse` | Declare deterministic snapshot reuse | See [reuse declarations](../../docs/Universal-Experiment-Loop.md#reuse-declarations) |

## Adjacent modules

- `limen.sfd` supplies the experiment configuration that this package runs.
- `limen.data` supplies raw data when a manifest declares a data source.
- `limen.log` analyzes completed runs.
- `limen.cohort` builds decoder cohorts on top of finished experiment results.
- `limen.inference` owns `Trainer`, `Sensor`, and `ReconstructionError`; the top-level `limen` package lazily re-exports those names.

## Quick orientation

| Module or family | Responsibility |
|---|---|
| `experiment_core.py` | UniversalExperimentLoop |
| `manifest_core.py` | Manifest and builder components |
| `checkpoint_manager.py` | Persist and restore advanced-run state |
| `feedback_controller.py` | Mid-run adaptive callbacks |
| `msq.py` | Mutable Search Queue |
| `param_domain.py` | Mutable parameter domain |
| `pruning_strategy.py` | Pruning interfaces and implementations |
| `param_search/` | SearchStrategy, GridStrategy, RandomStrategy |

<!-- reference: package-experiment-layout source=limen/experiment/README.md region=layout:package role=layout -->
```text
experiment/
├── README.md
├── __init__.py
├── _acceptance_settings.py
├── _allocation.py
├── _allocation_folds.py
├── _backtest_provenance.py
├── _boolean_controls.py
├── _column_shims.py
├── _dynamic_barriers.py
├── _factorize.py
├── _factorized_admission.py
├── _factorized_contract.py
├── _factorized_key.py
├── _factorized_native.py
├── _factorized_prep.py
├── _factorized_projection.py
├── _factorized_replay.py
├── _family_evidence.py
├── _filter_resume.py
├── _fixed_trade_sources.py
├── _fold_evidence.py
├── _fold_fingerprint.py
├── _fold_partitions.py
├── _fold_preparation.py
├── _grid_resume.py
├── _holdout_assessment.py
├── _holdout_attempts.py
├── _holdout_checkpoint.py
├── _holdout_composition.py
├── _holdout_events.py
├── _holdout_folds.py
├── _holdout_freeze.py
├── _holdout_history.py
├── _holdout_report.py
├── _holdout_research.py
├── _holdout_source.py
├── _holdout_sources.py
├── _manifest_preparation.py
├── _objective.py
├── _objective_optimizer.py
├── _objective_run.py
├── _parallel_commit.py
├── _parallel_resume.py
├── _parallel_rounds.py
├── _parallel_sources.py
├── _parallel_worker.py
├── _prepare_backtest_data.py
├── _prepare_trade_context.py
├── _recording.py
├── _required_columns.py
├── _resolve_backtest_config.py
├── _resolve_trade_policy.py
├── _result_fields.py
├── _round_signals.py
├── _round_storage.py
├── _serial_resume.py
├── _shim_fractional_diff.py
├── _source_contract.py
├── _validation_calibration.py
├── _validation_decisions.py
├── _walk_forward_run.py
├── _walk_forward_split.py
├── acceptance_report.py
├── checkpoint_manager.py
├── convergence_report.py
├── errors.py
├── experiment_core.py
├── family_acceptance_report.py
├── feedback_controller.py
├── holdout.py
├── manifest_core.py
├── msq.py
├── param_domain.py
├── param_search/
└── reducer/
```

## Things to know

- CLI and Python/MSQ runs share UEL with `SearchStrategy`, `ParamDomain`, and MSQ; checkpoints and feedback are optional.
- The ordinary MSQ path prepares every candidate. Omitting `search_strategy` retains the [legacy execution contract](../../docs/Universal-Experiment-Loop.md#legacy-execution-reference), including `ParamSpace` and the manifest requirement for `prep_each_round=True`.
- When `experiment_dir` is set on the MSQ path, Limen writes results and round data there; checkpoints and audit entries appear when their configured triggers run.
- Results are appended incrementally during a run, so experiment files should be treated as durable artifacts rather than temporary output.

## Read next

- [Universal Experiment Loop](../../docs/Universal-Experiment-Loop.md)
- [Advanced Search](../../docs/Advanced-Search.md)
- [Reducers And Feedback](../../docs/Reducers-And-Feedback.md)
- [Experiment Manifest](../../docs/Experiment-Manifest.md)

### Checkpoint protection evidence

Project-bound research checkpoints retain an explicit version-1 protection header in both `checkpoint.json` and `metadata.json`, including a null cutoff when no reservation is active. Their saved population retains source evidence, ordered terminal occurrences, and the completed rounds' original source, selection, split, schema, and SQL-history witnesses. Serial and parallel restoration validate this evidence before restoring state or rewriting artifacts. This private prerequisite does not enable public protected resume or parallel execution.

Trainer requires genuine completed protected experiment lineage before fitting. Valid earlier version-1 protected selections do not need a retroactive checkpoint header. For a legacy Python manifest whose actual local source differs from its saved YAML, SQL-validated source evidence supplies the local-source declaration for reconstruction; the reloaded admitted data and split geometry must still match exactly. The derived Sensor reference reflects that source without rewriting the saved experiment artifacts.
