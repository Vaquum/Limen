# Universal Experiment Loop

The Universal Experiment Loop (UEL) is Limen's execution engine. The normal operator path reaches it through `limen run`: YAML is validated, compiled into a manifest-backed SFD, executed by UEL, and written to a result directory.

This page covers:

- how CLI execution maps to UEL
- what `UniversalExperimentLoop` stores after a direct Python run
- when to use direct standard UEL versus the artifact-backed path
- which runtime rules matter for manifest-driven and custom SFDs

## Prerequisites

- a validated YAML manifest for the CLI path, or an SFD plus compatible data for direct Python use
- `prep_each_round=True` for manifest-driven SFDs
- an `experiment_dir` and `SearchStrategy` for checkpointed advanced runs

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

On the standard path, `random_search=False` enumerates the first `n_permutations` combinations in declared parameter order, with the first parameter varying fastest. Random search retains legacy sampling without exposing a seed; module-global `random.seed(...)` does not pin it. Use direct `ParamSpace(seed=...)` helper calls for seeded sampling.

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

Post-processing retains:

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
| `post_processing` | compute terminal post-run metrics (`uel._log`, confusion metrics, backtest results) |
| `progress_bar` | render the experiment progress bar; on by default, disable for headless runs |
| `record_execution` | persist test snapshot series and unscaled market returns; off by default; requires `search_strategy` and `experiment_dir` |
| `record_model_outputs` | persist test probabilities and boosting iteration counts; off by default; requires `search_strategy` and `experiment_dir` |

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

Primary attributes are listed below. `uel.data`, `uel.params`, `uel.experiment_log`, and `uel._log` are available after every successful run. Round artifact collections are retained only when `run(..., post_processing=True)` (or the corresponding advanced-run option) is enabled.

| Attribute | Meaning |
|---|---|
| `uel.data` | dataframe used by the run |
| `uel.params` | parameter space in use |
| `uel.round_params` | actual parameter values retained for each successful round when post-processing is enabled |
| `uel.experiment_log` | main round-by-round experiment log |
| `uel.experiment_confusion_metrics` | confusion-style analysis derived from predictions |
| `uel.experiment_backtest_results` | backtest-style analysis derived from predictions |
| `uel.preds` | test predictions retained when post-processing is enabled |
| `uel.scalers` | fitted scalers retained when post-processing is enabled |
| `uel._alignment` | alignment metadata retained when post-processing is enabled |
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

### Inline metrics in the round log

The `confusion_*` and `backtest_*` columns in `uel.experiment_log` and `results.csv` are produced **inline** — computed once per round inside the architecture's `evaluate()` while the round runs, not by post-run processing. `evaluate(data, inline_metrics=True)` appends them to the metrics dict the model returns, and UEL merges that dict verbatim into the round row. UEL forwards no flag of its own; on the manifest-driven path the architecture wrapper pins `inline_metrics=True`, so these columns are always present.

This is distinct from the two dedicated post-run frames. `uel.experiment_confusion_metrics` and `uel.experiment_backtest_results` are separate DataFrames built by the `Log` layer during finalization, only when `post_processing=True`. On deterministic, collision-free shipped ML evaluation paths, they carry the same values as the inline columns — the per-round `experiment_log['backtest_pnl_bps_p50']` equals the post-run `experiment_backtest_results['pnl_bps_p50']` — but as standalone frames rather than columns on the round log.

The inline confusion and backtest returns are price-gated: confusion counts always appear, but the price-derived `confusion_*_mean_return_pct` and `backtest_*` metrics are computed only when `price_data_for_backtest` is present for the round. Configured TP/SL requires valid prices and provenance before training, even when the current candidate resolves to null; unavailable prices raise. See [Reference Architecture](Reference-Architecture.md) for the `evaluate()` contract and the exact keys each mode adds.

TP/SL candidates belong to `sfd.manifest.backtest` references and ordinary search-parameter lists. Each candidate repeats preparation, training, and prediction; an exit-only sweep preserves predictions only with deterministic entry behavior. Rule-based entry architectures evaluate the fixed exits per split, while UEL still omits their post-run aggregate backtest table. Configured replay depends on in-memory provenance and is unavailable for file-only/resumed runs; Trainer rebuilds a fresh round. [Backtest](Backtest.md#tunable-take-profit-and-stop-loss) defines the canonical contract.

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
| `round_data.jsonl` | round params, predictions, alignment metadata, and optional execution and market returns |
| `checkpoint.json` | checkpoint state for resumption |
| `audit.jsonl` | feedback-controller audit trail |
| `interventions.json` | optional external intervention file polled by the feedback controller when the file exists |
| `metadata.json` | experiment metadata used by `Trainer` |

This path is what powers checkpointing, resumability, and the [Trainer](Trainer.md) workflow.

### Record execution

Set `uel.record_execution: true` in YAML, or pass `record_execution=True` to `run()` on the artifact-backed path. This is independent of `post_processing`. Each successful round gains `execution` in `round_data.jsonl`: full-precision `pos`, `gross`, and `net` arrays in test-row order, each multiplied once by the resolved `notional_rate`. Positions include execution lag and exits; regressors may backtest directional signals rather than their continuous saved predictions. Rule-based strategies record only test execution.

The same flag records the sibling `market: {"ret": [...]}`: unscaled returns from the original aligned test prices, in the same order and length as `execution.pos`. Each value is `close[i] / close[i-1] - 1` since the previous retained test row; row gaps can therefore span several source bars. The first value is JSON `null`, since the evaluated window supplies no predecessor. A value is also `null` when the original open, close, close-minus-open or previous close is NaN, the previous close is zero, or the computed return is nonfinite. This uses price tradability, independently of signals, execution lag, costs, notional and TP/SL settings; it is not a finite-positive-price mask.

When no snapshot runs (missing prices, disabled inline metrics, a custom producer, or event execution), both `execution` and `market` are `null`. A flat snapshot has full-length zero execution arrays and still records market returns. Event execution retains its existing `trade_ledger`. With recording off, neither field is added.

Read a new-format snapshot record with available market returns and reproduce its snapshot metrics:

```python
import json
import numpy as np
from limen.backtest.long_flat_strategy import ExecutionResult
from limen.backtest._snapshot_ledger import snapshot_ledger

with open("results/round_data.jsonl") as rows:
    record = json.loads(next(rows))
execution = record["execution"]
result = ExecutionResult(**{key: np.asarray(value) for key, value in execution.items()})
metrics = snapshot_ledger(result, 1.0)

ret = np.asarray(record["market"]["ret"], dtype=float)  # JSON null becomes NaN
valid = np.isfinite(ret) & np.isfinite(result.pos) & np.isfinite(result.gross)
timing_per_row = (
    result.gross[valid].mean()
    - result.pos[valid].mean() * ret[valid].mean()
)
```

The replayed metrics match `backtest_*` columns (`*_test` for rule-based strategies); using the original notional again would scale twice. For the shipped snapshot strategy without TP/SL, `gross[i] == execution.pos[i] * market.ret[i]` on available market rows. TP/SL uses fill-based exit returns, so a residual against original market closes includes exit effects. The reading example computes arithmetic timing per row over one common, nonempty finite population using gross returns; it adds no Limen metric. Net returns include costs, and compounded market return is a separate quantity.

Ordinal halves use `k = n // 2`, H1 rows `[0:k)` and H2 rows `[k:n)`; an odd middle row belongs to H2. For `n >= 2` with valid prices and no interior nulls, compounding available returns gives H1 `close[k-1] / close[0] - 1` and H2 `close[n-1] / close[k-1] - 1`. H2 includes the move across the boundary. Skipping interior nulls does not guarantee those endpoint identities. Window lengths and split dates can differ between rounds; calendar comparisons still require row identity and alignment.

Recording adds four arrays in total, including one full-precision market array per round. Serialized size and decoded-object overhead vary with the returns, test-window length and round count; Trainer and Cohort load whole round records.

Python resume must pass the same flag; CLI resume forwards the saved YAML flag. Changing it raises before artifacts are rewritten. Older metadata without the flag means `false`. Earlier execution-only records remain unchanged when resumed; readers must treat a missing `market` as unavailable, while new snapshot records include it. No artifact migration is performed. Existing resume requirements, including complete successful round records through the checkpoint, still apply.

### Record model outputs

Set `uel.record_model_outputs: true` in YAML, or pass `record_model_outputs=True` to `run()`. Each successful round records `probs` in `round_data.jsonl`, in the same order as `preds`, with `optimal_threshold` and `threshold_rule`. These are the original test probabilities used for that evaluation, including calibration when enabled. Apply `>` for uncalibrated predictions and `>=` for the calibration/threshold path; the uncalibrated threshold is `0.5`. RandomBinary records its existing `0.1`/`0.9` surrogate scores, which encode its sampled predictions. One-class LightGBM fits retain their original scores and record a constant decision boundary: `0`/`>=` for class `1`, or `1`/`>` for class `0`. Architectures without probabilities record `probs: null`.

LightGBM and XGBoost add `best_iteration` to `results.csv`: the number of boosting iterations actually used for prediction. LightGBM uses its positive `best_iteration_`, otherwise `n_iter_`. Tree-based XGBoost uses its zero-based best iteration plus one, otherwise the fitted booster's round count. XGBoost `gblinear` uses the final fitted round count because it does not retain an earlier model. Counts are available with early stopping disabled and with `inline_metrics=False`.

The option is independent of `record_execution` and `post_processing`; leaving it off preserves existing outputs. Python resume must pass the same flag; CLI resume uses the effective setting saved in metadata, including Python overrides. Metadata records an enabled setting, and a changed setting rejects resume before artifacts are rewritten. Older metadata without the flag means `false`. Probability arrays increase JSONL storage with test-window length.

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

The budget reducer can trim the remaining queue during a feedback cycle, so the number of completed rows may be lower than the requested permutation budget. `results.csv` and `round_data.jsonl` track completed rounds; `audit.jsonl` records the intervention; checkpoints follow `checkpoint_interval`.

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
- Continue to [Trainer](Trainer.md) for artifact-backed reconstruction of finished rounds into sensors.
