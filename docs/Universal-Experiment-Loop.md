# Universal experiment loop

The Universal Experiment Loop (UEL) is Limen's execution engine. The normal operator path reaches it through `limen run`: YAML is validated, compiled into a manifest-backed SFD, executed by UEL, and written to a result directory.

This page covers:

- how CLI execution maps to UEL
- what `UniversalExperimentLoop` stores after a direct Python run
- when to use direct standard UEL versus the artifact-backed path
- which runtime rules matter for manifest-driven and custom SFDs

## Preferred execution path

Start with CLI unless you are extending the engine directly:

```bash
limen validate logreg-first.yaml
limen profile logreg-first.yaml
limen run --dry-run logreg-first.yaml
limen run logreg-first.yaml
```

`limen run` constructs UEL with a compiled SFD, a concrete search strategy, an `experiment_dir`, and the parsed YAML stored as `yaml_reference` in `metadata.json`. The result directory contains the copied manifest, `metadata.json`, `results.csv`, and `round_data.jsonl`.

## Direct Python execution modes

Direct UEL integration currently has two execution modes.

| Mode | Entry path | Fits | Outputs |
|---|---|---|---|
| standard run path | instantiate with `sfd=` and optionally `data=`, then call `run()` without a `search_strategy` | custom local sweeps and Python examples | in-memory UEL artifacts plus a streaming CSV at `<experiment_name>.csv`, or `<experiment_dir>/<experiment_name>.csv` when `experiment_dir` is set |
| MSQ / artifact-backed path | instantiate with a concrete `search_strategy`, optionally `experiment_dir`, then call `run()` | advanced search flows, checkpointing, resumability, trainer workflows, and the CLI YAML path | `results.csv`, `round_data.jsonl`, checkpoints, audit trail, metadata, and in-memory UEL artifacts |

The standard run path is for direct Python work. The artifact-backed path is the durable engine path used by CLI YAML runs and advanced search.

The standard run path samples legacy `ParamSpace` combinations without exposing a seed; module-global `random.seed(...)` does not pin that sampling. Use direct `ParamSpace(seed=...)` helper calls when seeded legacy sampling is required.

## Direct standard run

This local Python example uses the file-backed spot-kline path with explicit `kline_size` and `row_count_limit`.

```python
import limen
from limen.data import HistoricalData

historical = HistoricalData()
historical.get_spot_klines(kline_size=7200, row_count_limit=2000)

uel = limen.UniversalExperimentLoop(
    data=historical.data,
    sfd=limen.sfd.logreg_binary,
)

uel.run(
    experiment_name='logreg-first',
    n_permutations=4,
    prep_each_round=True,
    random_search=False,
    post_processing=True,
)
```

With `post_processing=True`, these attributes are available:

```python
uel.experiment_log
uel.experiment_confusion_metrics
uel.experiment_backtest_results
```

Without `post_processing=True`, standard UEL still writes `uel.experiment_log`, but `uel._log`, `uel.experiment_confusion_metrics`, and `uel.experiment_backtest_results` remain unset.

On a live local run over that file-backed input with post-processing enabled, that produced:

- `uel.experiment_log` with one row per round
- `uel.experiment_confusion_metrics` with one row per round
- `uel.experiment_backtest_results` with one row per round
- `uel.preds`, `uel.round_params`, and `uel._alignment` for round-level reconstruction

## Constructor contract

```python
uel = limen.UniversalExperimentLoop(
    data=None,
    sfd=my_sfd,
    search_strategy=None,
    experiment_dir=None,
)
```

### Core constructor arguments

| Argument | Meaning |
|---|---|
| `sfd` | required SFD module |
| `data` | optional input dataframe; required for custom SFDs, optional for manifest-driven SFDs |
| `search_strategy` | advanced search hook; enables the MSQ execution path |
| `experiment_dir` | optional directory for stored run outputs; standard runs write their CSV there, and advanced runs write their artifact set there |
| `pruning_strategies`, `feedback_interval`, `checkpoint_interval`, `intra_callback` | advanced MSQ controls |
| `yaml_reference` | optional parsed YAML dict stored verbatim in `metadata.json` for reproducibility |

### Data behavior

- If the SFD exposes `manifest()` and `data=` is omitted, UEL fetches data from the manifest.
- If the SFD is custom and has no manifest, `data=` is required.
- For manifest-driven SFDs, the data source used is `fetch_data()` by default; pass `test_mode=True` to use the test data source.

## `run()` contract

```python
uel.run(
    experiment_name='my_experiment',
    n_permutations=100,
    prep_each_round=True,
)
```

### Core run arguments

| Argument | Meaning |
|---|---|
| `experiment_name` | run name and CSV path stem; `my_experiment` writes `my_experiment.csv`, or `experiment_dir/my_experiment.csv` when `experiment_dir` is set on the standard path |
| `n_permutations` | positive integer number of rounds to execute; YAML validation rejects bool, zero, negative, and values larger than the available parameter space |
| `prep_each_round` | whether prep runs every round; required for manifest-driven SFDs |
| `random_search` | random versus deterministic parameter generation on the standard path |
| `context_params` | extra static keys injected into every round |
| `params`, `prep`, `model` | optional overrides for the standard path |
| `resume` | resume from checkpoint in the advanced path |

### Manifest-driven rules

If the SFD uses `manifest()`:

- `prep_each_round=True` is required
- `prep=` and `model=` overrides are not allowed
- `params=` override is allowed

### Custom-SFD rules

If the SFD uses custom `prep()` and `model()`:

- `data=` must be provided when UEL is instantiated
- `prep_each_round` can be `True` or `False`, depending on whether prep depends on round params
- `params=`, `prep=`, and `model=` overrides are available on the standard path

## What UEL stores after a run

Primary attributes are:

| Attribute | Meaning |
|---|---|
| `uel.data` | dataframe used by the run |
| `uel.params` | parameter space in use |
| `uel.round_params` | actual parameter values used for each round |
| `uel.experiment_log` | main round-by-round experiment log |
| `uel.experiment_confusion_metrics` | confusion-style analysis derived from predictions |
| `uel.experiment_backtest_results` | backtest-style analysis derived from predictions |
| `uel.preds` | stored test predictions per round |
| `uel.scalers` | fitted scalers captured from prep or manifest scaling |
| `uel._alignment` | alignment metadata per round |
| `uel._log` | internal `Log` object for deeper analysis |

### Alignment metadata

Each entry in `uel._alignment` includes:

- `missing_datetimes`
- `first_test_datetime`
- `last_test_datetime`

This is what lets downstream analysis stay aligned with the actual test window seen by a round.

### Deeper post-run analysis

UEL constructs a `Log` instance automatically at the end of a successful run. That exposes methods such as:

- `uel._log.permutation_prediction_performance(round_id=0)`
- `uel._log.permutation_confusion_metrics('price_change', round_id=0)`
- `uel.experiment_parameter_correlation('auc')`

## Standard path versus artifact-backed path

### Standard path

The standard path writes a streaming CSV at:

```text
<experiment_name>.csv
```

When `experiment_dir` is set, the standard path writes:

```text
<experiment_dir>/<experiment_name>.csv
```

and keeps the full run state in memory on the `uel` object.

This is the path to use for:

- direct local research loops
- custom Python examples
- direct parameter sweeps

### Artifact-backed path

When UEL is instantiated with a concrete `search_strategy` and an `experiment_dir`, Limen stores structured artifacts there. This path uses `results.csv` as the round log filename rather than `<experiment_name>.csv`.

| File | Meaning |
|---|---|
| `results.csv` | streaming round log; if a round fails a `strict_mode` null check, a `strict_mode_error` column records the error message and all metric columns for that round are empty |
| `round_data.jsonl` | round params, predictions, and alignment metadata |
| `checkpoint.json` | checkpoint state for resumption |
| `audit.jsonl` | feedback-controller audit trail |
| `interventions.json` | optional external intervention file polled by the feedback controller when the file exists |
| `metadata.json` | experiment metadata used by `Trainer` |

This path is what powers checkpointing, resumability, and the [Trainer](Trainer.md) workflow.

### Important scope note

Limen ships built-in strategies (`GridStrategy`, `RandomStrategy`) and the `SearchStrategy` abstraction for custom strategies. The advanced path is available with built-in strategies or a custom implementation from the caller's codebase.

## One real advanced run

The UEL-facing part of an advanced run looks like this with a concrete `SearchStrategy`:

```python
import limen

from limen.experiment.param_domain import ParamDomain
from limen.experiment.reducer import BudgetReducer

domain = ParamDomain(limen.sfd.random_binary.params())
strategy = MiniGrid(domain)  # see Advanced Search for a complete minimal implementation

uel = limen.UniversalExperimentLoop(
    sfd=limen.sfd.random_binary,
    search_strategy=strategy,
    pruning_strategies=[
        BudgetReducer(max_permutations=4, check_after_pct=0.25),
    ],
    feedback_interval=2,
    checkpoint_interval=3,
    experiment_dir='advanced-budget',
)

uel.run(
    experiment_name='advanced-budget',
    n_permutations=6,
)
```

On a live local run in this repo, that advanced run:

- requested `6` permutations
- finished with `4` rows in `results.csv`
- wrote `4` entries to `round_data.jsonl`
- wrote `1` entry to `audit.jsonl`
- saved a checkpoint after round `3`

That behavior came from a reducer-triggered trim during the feedback cycle, not an early stop.

## Resume in practice

Resumption belongs only to the advanced path:

```python
uel.run(
    experiment_name='advanced-budget',
    n_permutations=6,
    resume=True,
)
```

In a live shutdown-and-resume run in this repo:

- the first phase stopped after `2` completed rounds
- `results.csv` and `round_data.jsonl` each contained `2` entries
- the resumed phase finished the remaining rounds
- the final stored round ids were `0, 1, 2, 3`

Use the same `experiment_dir`, strategy type, and reducer configuration when resuming.

For the full advanced-search contract, continue to [Advanced Search](Advanced-Search.md) and [Reducers And Feedback](Reducers-And-Feedback.md).

## Common Errors

### Manifest-driven runs require `prep_each_round=True`

A manifest-driven SFD with `prep_each_round=False` raises `prep_each_round must be True for manifest-driven SFDs`. Set `prep_each_round=True`.

### Manifest-driven runs cannot override `prep` or `model`

Passing `prep=` or `model=` to `run()` for a manifest-driven SFD raises `Cannot override prep/model when SFD has manifest`. Put the logic into the manifest, or switch to the custom SFD path.

### Custom SFDs require explicit `data=`

A custom SFD with omitted `data=` raises `data parameter required for custom SFDs using custom functions approach`.

### Resuming requires a search strategy

Resumption belongs to the advanced path. Calling `run(resume=True)` without a search strategy raises `resume=True is only supported with a search_strategy`.

## Read next

- Continue to [Log](Log.md) to understand the analysis surfaces built on top of UEL results.
- Continue to [Experiment Manifest](Experiment-Manifest.md) for manifest-driven SFD construction.
- Continue to [Trainer](Trainer.md) for artifact-backed retraining of finished rounds into sensors.
