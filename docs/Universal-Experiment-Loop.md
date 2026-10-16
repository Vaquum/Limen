# Universal Experiment Loop

The Universal Experiment Loop (UEL) executes Limen experiments. Choose the CLI for YAML manifests or the Python API for SFDs and direct search control. Both routes use the Mutable Search Queue (MSQ) inside UEL: a search strategy generates candidates, MSQ controls which candidate runs next, and UEL prepares data and evaluates it.

This page covers:

- how CLI execution maps to UEL
- what `UniversalExperimentLoop` stores after a direct Python run
- how to run a Python SFD with a built-in search strategy
- how to enable persistence, resume, and adaptive feedback
- which runtime rules matter for manifest-driven and custom SFDs

## Prerequisites

- `pip install "vaquum-limen[data]"` for the example below and access to its market-data source
- a validated YAML manifest for the CLI path, or an SFD plus compatible data for direct Python use
- a `SearchStrategy` for Python/MSQ execution
- an `experiment_dir` when you need persisted round artifacts or checkpoints

## Preferred execution path

Start with CLI for a first YAML experiment:

```bash
limen validate logreg-first.yaml
limen profile logreg-first.yaml
limen run --dry-run logreg-first.yaml
limen run logreg-first.yaml
```

`limen run` constructs UEL with a compiled SFD, a concrete search strategy, an `experiment_dir`, and the parsed YAML stored as `yaml_reference` in `metadata.json`. The result directory contains the copied manifest, `metadata.json`, `results.csv`, and `round_data.jsonl.gz`.

For trials evaluated across successive test windows, see [walk-forward sweeps](Experiment-Manifest.md#walk-forward-sweeps). Each trial runs all configured folds before its aggregate result enters the existing feedback loop.

## Direct Python run

Use Python when the experiment is expressed as an SFD, when preparation or models need custom code, or when you need direct search control. A custom SFD exposes `params()`, `prep()`, and `model()` and receives `data=` explicitly; it does not need a YAML manifest or a built-in model architecture.

Pass `search_strategy` explicitly to select MSQ. The constructor still defaults to `None`, which selects the supported [standard execution contract](#legacy-execution-reference).

This example uses a built-in SFD and seeded random search. Replace the SFD with your own module to run custom code under the same loop.

```python
import limen
from limen.data import HistoricalData
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import RandomStrategy

historical = HistoricalData()
data = historical.get_spot_klines(kline_size=7200, row_count_limit=2000)
sfd = limen.sfd.logreg_binary
strategy = RandomStrategy(ParamDomain(sfd.params()), seed=42)

uel = limen.UniversalExperimentLoop(
    data=data,
    sfd=sfd,
    search_strategy=strategy,
)

uel.run(
    experiment_name='logreg-first',
    n_permutations=4,
    post_processing=True,
)
```

With `post_processing=True`, these attributes are available:

```python
uel.experiment_log
uel.experiment_confusion_metrics
uel.experiment_backtest_results
```

Without `post_processing=True`, UEL still populates `uel.experiment_log`, but `uel._log`, `uel.experiment_confusion_metrics`, and `uel.experiment_backtest_results` remain unset. With no `experiment_dir`, the run writes `logreg-first.csv`; it does not persist round data or checkpoints.

For exhaustive search, use `GridStrategy` over the same `ParamDomain`. The random-search seed controls candidate generation, not randomness inside preparation or model code. See [Advanced Search](Advanced-Search.md) for strategy choices and optional feedback.

The [convergence warning report](#convergence-warning-report) is available after run conclusion independently of `post_processing`.

Post-processing retains:

- `uel.experiment_log` with one row per round
- `uel.experiment_confusion_metrics` with one row per round
- `uel.experiment_backtest_results` with one row per round
- `uel.preds`, `uel.round_params`, and `uel._alignment` for round-level reconstruction

## Constructor contract

```python
uel = limen.UniversalExperimentLoop(
    data=data,
    sfd=sfd,
    search_strategy=strategy,
    experiment_dir=None,
)
```

### Core constructor arguments

| Argument | Meaning |
|---|---|
| `sfd` | required SFD module |
| `data` | optional input dataframe; required for custom SFDs, optional for manifest-driven SFDs |
| `search_strategy` | candidate-generation strategy; selects the MSQ execution path |
| `experiment_dir` | optional directory for saved results, round data, metadata, and checkpoints |
| `pruning_strategies`, `feedback_interval`, `checkpoint_interval`, `intra_callback` | optional feedback and checkpoint controls |
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
)
```

### Core run arguments

| Argument | Meaning |
|---|---|
| `experiment_name` | run name; without `experiment_dir`, writes `<experiment_name>.csv`; with it, writes `results.csv` there |
| `n_permutations` | positive integer number of rounds to execute; YAML validation rejects bool, zero, negative, and values larger than the available parameter space |
| `context_params` | extra static keys injected into every round |
| `resume` | restore a checkpoint from `experiment_dir` |
| `post_processing` | compute terminal post-run metrics (`uel._log`, confusion metrics, backtest results) |
| `progress_bar` | render the experiment progress bar; on by default, disable for headless runs |
| `record_execution` | persist test snapshot series and unscaled market returns; off by default; requires `search_strategy` and `experiment_dir` |
| `record_model_outputs` | persist test probabilities and boosting iteration counts; off by default; requires `search_strategy` and `experiment_dir` |
| `factorize` | opt-in reuse of fitted preparation state across model variations, and predictions across independent snapshot economics, in a deterministic persisted MSQ grid; default `false` |

### Factorized snapshot grids

Set `uel.factorize: true` in YAML or `run(..., factorize=True)`. The existing grid still emits one ordinary result row per full parameter combination, with the same backtest executor and numeric results. This single switch enables two reuse boundaries:

- Across model variations with identical preparation inputs, reuse the fitted target and, when configured, the built-in `RobustScaler` statistics. Generate features and apply transformations again; fit and predict each distinct model afresh.
- Across independent backtest-only variations, reuse the resulting predictions and recording evidence. Run the existing snapshot executor for every economic combination.

Omitted or `false` retains fresh preparation, model fitting and prediction for every permutation, including repeated identical parameters. Default execution remains sequential. Factorization changes which computations repeat, not the experiment's round order or full-permutation identities.

The switch supports unshuffled grids with independent backtest-only axes, built-in deterministic DLinear, deterministic single-thread LightGBM, or shipped rule-based snapshot architectures. Fitted preparation reuse is limited to the already supported deterministic targets and a statically declared built-in `RobustScaler`; dynamic/custom scalers and PCA remain unsupported. A change to any preparation dependency requires a fresh fit, including a model parameter also referenced by a feature, target or scaler declaration. Prepared frames and fitted models are never retained in the preparation cache.

Ambiguous dependencies, stochastic settings, callbacks, pruning, event execution, calibration, walk-forward, custom architectures and `workers > 1` fail explicitly before UEL artifact writes. Context overrides pass the same admission checks; intervention files and candidates outside the admitted domain reject. Factorization does not silently substitute ordinary execution.

Both caches belong to one sweep invocation and contain only successful work. Fitted artifacts are copied before use by another round. The prediction cache retains predictions, price/provenance and recording evidence; both caches grow with distinct keys and have no eviction. Large grids may require ordinary execution. Log post-processing and resumed invocations retain ordinary reconstruction work. CLI resume reads the effective setting from metadata; caches restart empty. Changing YAML `factorize` changes canonical manifest identity but never full-permutation round IDs.

The recorded-data benchmark isolates the fitted-preparation extension: both campaigns use `factorize=True` and retain prediction reuse; the baseline clears only the fitted-preparation cache before each evaluation. It records actual target/scaler fits, stage timings, end-to-end elapsed time, peak process RSS and scientific-output digests. It uses fixed 2024 hourly observations, pinned splits and single-threaded LightGBM settings; its bounded workload makes no universal speedup claim.

```bash
python -m tests.benchmark_factorized_preparation --output /tmp/limen-factorized-preparation
```

### SFD execution contract

On the ordinary MSQ path, UEL calls the SFD's preparation and model functions for each candidate. Manifest-driven SFDs supply these functions through their manifest. Custom SFDs supply them directly and require explicit `data=`. Configure candidate values in the strategy's `ParamDomain`; configure custom functions in the SFD. The `factorize` option above has a separate, narrower manifest-only contract.

When `search_strategy` is set, the legacy-only `run()` arguments `prep_each_round`, `random_search`, `maintain_details_in_params`, `params`, `prep`, and `model` are accepted but ignored. Configure the strategy and SFD instead.

For saved custom runs, the SFD must expose `__name__` and preparation must supply the expected `_alignment` metadata. Arbitrary custom SFD execution does not imply Trainer compatibility: Trainer reconstructs a manifest from a saved `yaml_reference`.

## What UEL stores after a run

Primary attributes are listed below. `uel.data`, `uel.params`, and `uel.experiment_log` are available after every successful run. `uel._log` and round artifact collections are retained only when `run(..., post_processing=True)` is enabled.

| Attribute | Meaning |
|---|---|
| `uel.data` | dataframe used by the run |
| `uel.params` | parameter space in use |
| `uel.round_params` | actual parameter values retained for each successful round when post-processing is enabled |
| `uel.experiment_log` | main round-by-round experiment log |
| `uel.convergence_report` | recorded convergence warning counts, rates and parameter-value summaries after run conclusion |
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

### Convergence warning report

After a completed standard, MSQ or walk-forward run, read:

```python
report = uel.convergence_report
report["convergence_warning_rounds"]
report["observed_rounds"]
report["convergence_warning_pct"]
```

This dictionary reports explicitly recorded `ConvergenceWarning` categories, including subclasses. Warning message text is not used to infer convergence problems. A recorded absence of that warning means no warning was observed; it does not prove the model converged.

`convergence_warning_pct` is 100 times `convergence_warning_rounds` divided by `observed_rounds`: finished rounds with available diagnostics. Missing diagnostics, older records without the category observation, and strict-mode failed rounds are unavailable and excluded from that denominator. `rounds` and `unavailable_rounds` list the total and unavailable counts separately. With no observed rounds, the percentage is `None` in memory and JSON `null`.

For walk-forward runs, each trial counts once: a warning in any observed fold makes the trial warning-bearing. A trial can be recorded without a warning only when every fold has available diagnostics and none records one; otherwise its observation is unavailable.

`parameter_patterns` lists each parameter value associated with at least one recorded warning. Each entry carries `parameter`, a display string `value`, `observed_rounds`, `convergence_warning_rounds` and `convergence_warning_pct`. Its denominator includes every observed round with that value, including those without warnings. These are marginal descriptive frequencies, not evidence that a parameter caused a warning; combinations can be confounded and groups can overlap across parameters.

UEL computes this report even with `post_processing=False`. With `experiment_dir` configured, it writes `convergence_report.json` there. Otherwise it retains the dictionary in memory. Both paths log the overall counts and rate at run conclusion.

### Deeper post-run analysis

With `post_processing=True`, UEL constructs a live `Log` at run conclusion. It supports explicit retained row analysis. Captured aggregates can also be read through `limen.Log(uel_object=uel)` without that option, or through a CSV-backed `Log`; see the [canonical aggregate and row contracts](Log.md). The retained path exposes methods such as:

- `uel._log.permutation_prediction_performance(round_id=0)`
- `uel._log.permutation_confusion_metrics('price_change', round_id=0)`
- `uel.experiment_parameter_correlation('auc')`

### Inline metrics in the round log

The `confusion_*` and `backtest_*` columns in `uel.experiment_log` and `results.csv` are produced **inline** — computed once per round inside the architecture's `evaluate()` while the round runs, not by post-run processing. `evaluate(data, inline_metrics=True)` appends them to the metrics dict the model returns, and UEL merges that dict verbatim into the round row. UEL forwards no flag of its own; on the manifest-driven path the architecture wrapper pins `inline_metrics=True`, so these columns are always present.

With `post_processing=True`, `uel.experiment_backtest_results` reads captured backtest columns through `Log`, while `uel.experiment_confusion_metrics` keeps the existing explicit `price_change` row summary. The latter applies trimming and additional statistics; it is distinct from the untrimmed captured confusion view returned by `Log.experiment_confusion_metrics()` with no argument. [Log](Log.md) defines their values, units, evidence requirements and walk-forward fold behavior.

The inline confusion and backtest returns are price-gated: confusion counts always appear, but the price-derived `confusion_*_mean_return_pct` and `backtest_*` metrics are computed only when `price_data_for_backtest` is present for the round. Configured TP/SL requires valid prices and provenance before training, even when the current candidate resolves to null; unavailable prices raise. See [Reference Architecture](Reference-Architecture.md) for the `evaluate()` contract and the exact keys each mode adds.

TP/SL candidates belong to `sfd.manifest.backtest` references and ordinary search-parameter lists. Each candidate repeats preparation, training, and prediction; an exit-only sweep preserves predictions only with deterministic entry behavior. Rule-based entry architectures evaluate the fixed exits per split, while UEL still omits their post-run aggregate backtest table. Captured aggregate reads preserve the recorded execution result; explicit replay has separate source-evidence requirements. [Log](Log.md#backtest-surface) defines the aggregate contract, and [Backtest](Backtest.md#tunable-take-profit-and-stop-loss) defines the trading policy.

## Persisting Python/MSQ runs

When UEL is instantiated with a concrete `search_strategy` and an `experiment_dir`, Limen stores structured artifacts there. This path uses `results.csv` as the round log filename rather than `<experiment_name>.csv`.

| File | Meaning |
|---|---|
| `results.csv` | streaming round log; if a round fails a `strict_mode` null check, a `strict_mode_error` column records the error message and all metric columns for that round are empty |
| `round_data.jsonl.gz` | round params, predictions, alignment metadata, and optional execution and market returns |
| `checkpoint.json` | checkpoint state for resumption |
| `audit.jsonl` | feedback-controller audit trail |
| `interventions.json` | optional external intervention file polled by the feedback controller when the file exists |
| `metadata.json` | experiment metadata used by `Trainer` |
| `convergence_report.json` | convergence warning report written at run conclusion, independently of post-processing |

Saved state powers checkpointing and resumability. The CLI also stores the YAML definition in `metadata.json`; [Trainer](Trainer.md) and CLI resume require that reconstructible `yaml_reference`. A Python run with arbitrary custom functions and an `experiment_dir` does not automatically meet this requirement.

### Record execution

Set `uel.record_execution: true` in YAML, or pass `record_execution=True` to a Python/MSQ run with `experiment_dir`. This is independent of `post_processing`. Each successful round gains `execution` in `round_data.jsonl.gz`: full-precision `pos`, `gross`, and `net` arrays in test-row order, each multiplied once by the resolved `notional_rate`. Positions include execution lag and exits; regressors may backtest directional signals rather than their continuous saved predictions. Rule-based strategies record only test execution.

The same flag records the sibling `market: {"ret": [...]}`: unscaled returns from the original aligned test prices, in the same order and length as `execution.pos`. Each value is `close[i] / close[i-1] - 1` since the previous retained test row; row gaps can therefore span several source bars. The first value is JSON `null`, since the evaluated window supplies no predecessor. A value is also `null` when the original open, close, close-minus-open or previous close is NaN, the previous close is zero, or the computed return is nonfinite. This uses price tradability, independently of signals, execution lag, costs, notional and TP/SL settings; it is not a finite-positive-price mask.

When no snapshot runs (missing prices, disabled inline metrics, a custom producer, or event execution), both `execution` and `market` are `null`. A flat snapshot has full-length zero execution arrays and still records market returns. Event execution retains its existing `trade_ledger`. With recording off, neither field is added.

Read a new-format snapshot record with available market returns and reproduce its snapshot metrics:

```python
import json
from pathlib import Path
from limen.experiment._round_storage import open_text, round_path
import numpy as np
from limen.backtest.long_flat_strategy import ExecutionResult
from limen.backtest._snapshot_ledger import snapshot_ledger

with open_text(round_path(Path("results"))) as rows:
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

Python resume must pass the same flag; CLI resume uses the effective setting saved in metadata, including Python overrides. Changing it raises before artifacts are rewritten. Older metadata without the flag means `false`. Earlier execution-only records remain unchanged when resumed; readers must treat a missing `market` as unavailable, while new snapshot records include it. No artifact migration is performed. Existing resume requirements, including complete successful round records through the checkpoint, still apply.

### Record model outputs

Set `uel.record_model_outputs: true` in YAML, or pass `record_model_outputs=True` to `run()`. Each successful round records `probs` in `round_data.jsonl.gz`, in the same order as `preds`, with `optimal_threshold` and `threshold_rule`. These are the original test probabilities used for that evaluation, including calibration when enabled. Apply `>` for uncalibrated predictions and `>=` for the calibration/threshold path; the uncalibrated threshold is `0.5`. RandomBinary records its existing `0.1`/`0.9` surrogate scores, which encode its sampled predictions. One-class LightGBM fits retain their original scores and record a constant decision boundary: `0`/`>=` for class `1`, or `1`/`>` for class `0`. Architectures without probabilities record `probs: null`.

LightGBM and XGBoost add `best_iteration` to `results.csv`: the number of boosting iterations actually used for prediction. LightGBM uses its positive `best_iteration_`, otherwise `n_iter_`. Tree-based XGBoost uses its zero-based best iteration plus one, otherwise the fitted booster's round count. XGBoost `gblinear` uses the final fitted round count because it does not retain an earlier model. Counts are available with early stopping disabled and with `inline_metrics=False`.

The option is independent of `record_execution` and `post_processing`; leaving it off preserves existing outputs. Python resume must pass the same flag; CLI resume uses the effective setting saved in metadata, including Python overrides. Metadata records an enabled setting, and a changed setting rejects resume before artifacts are rewritten. Older metadata without the flag means `false`. Probability arrays increase JSONL storage with test-window length.

## Optional adaptive search

Optional reducers can adapt the search while UEL runs. This example adds a budget reducer to a built-in grid strategy:

```python
import limen

from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import GridStrategy
from limen.experiment.reducer import BudgetReducer

domain = ParamDomain(limen.sfd.random_binary.params())
strategy = GridStrategy(domain)

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

The budget reducer can trim the remaining queue during a feedback cycle, so the number of completed rows may be lower than the requested permutation budget. `results.csv` and `round_data.jsonl.gz` track completed rounds; `audit.jsonl` records the intervention; checkpoints follow `checkpoint_interval`.

## Resume in practice

Python/MSQ runs can resume when an `experiment_dir` and a checkpoint are available:

```python
uel.run(
    experiment_name='advanced-budget',
    n_permutations=6,
    resume=True,
)
```

In a live shutdown-and-resume run in this repo:

- the first phase stopped after `2` completed rounds
- `results.csv` and `round_data.jsonl.gz` each contained `2` entries
- the resumed phase finished the remaining rounds
- the final stored round indices were `0, 1, 2, 3`

Use the same `experiment_dir`, strategy type, and reducer configuration when resuming.

For strategy and feedback controls, continue to [Advanced Search](Advanced-Search.md) and [Reducers And Feedback](Reducers-And-Feedback.md).

## Legacy execution reference

UEL also retains the standard `ParamSpace` execution path for existing callers. Omitting `search_strategy` selects it. New examples use Python/MSQ; this reference records the legacy contract without changing its support.

| Legacy `run()` argument | Behavior |
|---|---|
| `random_search` | random sampling, or deterministic enumeration when `False` |
| `prep_each_round` | repeat preparation; `True` is required for manifest-driven SFDs; custom SFDs may prepare once when it is `False` |
| `params` | callable override for the parameter space |
| `prep`, `model` | callable overrides for custom SFDs; disallowed for manifest-driven SFDs |
| `maintain_details_in_params` | expose `_experiment_details` to the current round |

The path writes `<experiment_name>.csv`, or `<experiment_dir>/<experiment_name>.csv` when a directory is supplied. It uses integer round IDs and has no checkpoint/resume or MSQ feedback. `post_processing=True` enables the same live analysis attributes.

With `random_search=False`, combinations follow declared parameter order, with the first parameter varying fastest. Legacy UEL random sampling exposes no seed; module-global `random.seed(...)` does not pin it. The [ParamSpace utility](Utilities.md#paramspace) exposes its own seed for direct helper use. These ordering and sampling contracts differ from the MSQ strategies.

### Direct standard run

For existing callers, omit `search_strategy` and use the legacy arguments. This supported contract remains available; the example uses `data` and `sfd` from the Python example above:

```python
standard = limen.UniversalExperimentLoop(data=data, sfd=sfd)
standard.run(
    experiment_name='standard-run',
    n_permutations=4,
    prep_each_round=True,
    random_search=False,
    post_processing=True,
)
```

## Common Errors

### Legacy manifest-driven runs require `prep_each_round=True`

A manifest-driven SFD with `prep_each_round=False` raises `prep_each_round must be True for manifest-driven SFDs`. Set `prep_each_round=True`.

### Legacy manifest-driven runs cannot override `prep` or `model`

Passing `prep=` or `model=` to `run()` for a manifest-driven SFD raises `Cannot override prep/model when SFD has manifest`. Put the logic into the manifest, or switch to the custom SFD path.

### Custom SFDs require explicit `data=`

A custom SFD with omitted `data=` raises `data parameter required for custom SFDs using custom functions approach`.

### Resuming requires a search strategy

Calling `run(resume=True)` without a search strategy raises `resume=True is only supported with a search_strategy`. Python/MSQ resume also requires a saved checkpoint directory.

## Read next

- Continue to [Log](Log.md) to understand the analysis surfaces built on top of UEL results.
- Continue to [Experiment Manifest](Experiment-Manifest.md) for manifest-driven SFD construction.
- Continue to [Trainer](Trainer.md) for reconstruction of saved YAML experiment rounds into sensors.

## Round storage format

New result directories declare `round_storage: gzip-jsonl-v1` in metadata and write `round_data.jsonl.gz`. Each gzip member contains one UTF-8 JSON round plus a newline, compressed at fixed level 1 with mtime zero and no filename header. Walk-forward fold objects remain intact and receive gzip compression only. Existing directories without the discriminator retain plain `round_data.jsonl`, including on resume. Existing run/CLI signatures and recording flags do not change.

For ordinary rounds, a non-null top-level `market` is replaced in place by `market_ref`. Its lowercase SHA-256 digest addresses `round_market/<digest>.json.gz`. The decompressed blob bytes are exactly `json.dumps(market, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('utf-8')`; hash those bytes and verify them before reconstruction. Alignment stays in the owning round. Restore the market key at the reference's position; missing/null market stays missing/null. This preserves exact values, numeric types, signed zero and ordering without lossy conversion. Existing nonfinite tokens elsewhere in round/fold results retain the legacy JSON behavior.

External readers must verify each member trailer before exposing its round. The internal shared reader uses `zlib.decompressobj(wbits=31)`, requires each gzip header, rejects zero padding, and checks EOF/CRC/ISIZE. It validates references before constructing a path and verifies blob digest and return length. Decompression is bounded to 512 MiB per round and 128 MiB per market. The package-local example above uses the private shared reader; this is not a new public reader API.

Trainer, Cohort and acceptance processing reject incomplete artifacts. Live/resume reads may ignore only an incomplete final member outside the successful checkpoint prefix. Resume validates ordered successful CSV identities, scans existing member/line offsets, and truncates in place: committed round-data bytes and inode survive. CSV retains its existing rewrite behavior. This is sequential restoration, not indexed random access.

Blob publication precedes its referencing member through a same-directory temporary file and atomic replace. The contract covers process interruption, not new power-loss/fsync durability or multiple writers in one directory. Orphan blobs and temporary leftovers are tolerated, never used as alternate references, and included in storage accounting; no garbage collector is introduced. Unknown/conflicting layouts and integrity failures raise errors before successful resume or report replacement.

Compression reduces disk output; it does not change worker IPC, model execution or scientific results. Unique markets still require unique files and save less. Grid currently pins its Limen package version: update its reader before upgrading that separate dependency pin. No historical run is converted automatically.
