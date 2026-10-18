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
| `recording` | choose saved round-log payload groups; all disabled by default; explicit policies require `search_strategy` and `experiment_dir`; see [recording policy](#recording-policy) |
| `factorize` | opt-in reuse of fitted preparation state across model variations, and predictions across independent snapshot economics, in a deterministic persisted MSQ grid; default `false` |

### Factorized snapshot grids

Set `uel.factorize: true` in YAML or `run(..., factorize=True)`. The existing grid still emits one ordinary result row per full parameter combination, with the same backtest executor and numeric results. This single switch enables two reuse boundaries:

- Across model variations with identical preparation inputs, reuse the fitted target and, when configured, the built-in `RobustScaler` statistics. Generate features and apply transformations again; fit and predict each distinct model afresh.
- Across independent backtest-only variations, reuse the resulting predictions and recording evidence. Run the existing snapshot executor for every economic combination.

Omitted or `false` retains fresh preparation, model fitting and prediction for every permutation, including repeated identical parameters. Default execution remains sequential. Factorization changes which computations repeat, not the experiment's round order or full-permutation identities.

The switch supports unshuffled grids and [seeded version-2 shuffled grids](Advanced-Search.md#seeded-finite-traversal) with independent backtest-only axes, built-in deterministic DLinear, deterministic single-thread LightGBM, or shipped rule-based snapshot architectures. Fitted preparation reuse is limited to the already supported deterministic targets and a statically declared built-in `RobustScaler`; dynamic/custom scalers and PCA remain unsupported. A change to any preparation dependency requires a fresh fit, including a model parameter also referenced by a feature, target or scaler declaration. Prepared frames and fitted models are never retained in the preparation cache.

Ambiguous dependencies, stochastic settings, callbacks, pruning, event execution, calibration, walk-forward, custom architectures and `workers > 1` fail explicitly before UEL artifact writes. Context overrides pass the same admission checks; intervention files and candidates outside the admitted domain reject. Factorization does not silently substitute ordinary execution.

Both caches belong to one sweep invocation and contain only successful work. Fitted artifacts are copied before use by another round. The prediction cache retains predictions, price/provenance and recording evidence; each cache retains at most **128 entries**, evicting the least recently used entry. Hits refresh recency; evicted keys recompute on their next access. Prediction entries include copied fitted-preparation evidence, so this is 128 prediction entries plus 128 separate preparation entries, not 128 fitted objects total or a byte/RSS limit. Source observations, active-round work, domain/seen-set and result/log growth remain separate costs. Reuse fits a key once while resident; shuffled prefixes may provide few hits and eviction may repeat work without changing scientific outputs. Independent runs each own these costs. Log post-processing and resumed invocations retain ordinary reconstruction work. CLI resume reads the effective setting from metadata; caches restart empty. Strict-mode failures consume a terminal candidate, retain a nonempty failure marker and have no successful prediction record; recovery joins only successful occurrences. No automatic retry or recovery before the first existing checkpoint is added. Changing YAML `factorize` changes canonical manifest identity but never full-permutation round IDs.

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
| `round_data.jsonl.gz` | round identity and params, with payloads selected by the [recording policy](#recording-policy) and existing recording flags |
| `checkpoint.json` | checkpoint state for resumption |
| `audit.jsonl` | feedback-controller audit trail |
| `interventions.json` | optional external intervention file polled by the feedback controller when the file exists |
| `metadata.json` | experiment metadata used by `Trainer` |
| `convergence_report.json` | convergence warning report written at run conclusion, independently of post-processing |

Saved state powers checkpointing and resumability. The CLI also stores the YAML definition in `metadata.json`; [Trainer](Trainer.md) and CLI resume require that reconstructible `yaml_reference`. A Python run with arbitrary custom functions and an `experiment_dir` does not automatically meet this requirement.

### Record execution

Set `uel.record_execution: true` in YAML, or pass `record_execution=True` to a Python/MSQ run with `experiment_dir`. This is independent of `post_processing`. Each successful round gains `execution` in `round_data.jsonl.gz`: full-precision `pos`, `gross`, and `net` arrays in test-row order, each multiplied once by the resolved `notional_rate`. Positions include execution lag and exits; regressors may backtest directional signals rather than their continuous saved predictions. Rule-based strategies record only test execution.

The same flag records the sibling `market: {"ret": [...]}`: unscaled returns from the original aligned test prices, in the same order and length as `execution.pos`. Each value is `close[i] / close[i-1] - 1` since the previous retained test row; row gaps can therefore span several source bars. The first value is JSON `null`, since the evaluated window supplies no predecessor. A value is also `null` when the original open, close, close-minus-open or previous close is NaN, the previous close is zero, or the computed return is nonfinite. This uses price tradability, independently of signals, execution lag, costs, notional and TP/SL settings; it is not a finite-positive-price mask.

When no snapshot runs (missing prices, disabled inline metrics, a custom producer, or event execution), both `execution` and `market` are `null`. A flat snapshot has full-length zero execution arrays and still records market returns. Event execution retains its existing `trade_ledger` by default. With recording off, neither field is added.

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

Set `uel.record_model_outputs: true` in YAML, or pass `record_model_outputs=True` to `run()`. Each successful round records `probs` in `round_data.jsonl.gz`, in test-row order (matching `preds` when enabled), with `optimal_threshold` and `threshold_rule`. These are the original test probabilities used for that evaluation, including calibration when enabled. Apply `>` for uncalibrated predictions and `>=` for the calibration/threshold path; the uncalibrated threshold is `0.5`. RandomBinary records its existing `0.1`/`0.9` surrogate scores, which encode its sampled predictions. One-class LightGBM fits retain their original scores and record a constant decision boundary: `0`/`>=` for class `1`, or `1`/`>` for class `0`. Architectures without probabilities record `probs: null`.

LightGBM and XGBoost add `best_iteration` to `results.csv`: the number of boosting iterations actually used for prediction. LightGBM uses its positive `best_iteration_`, otherwise `n_iter_`. Tree-based XGBoost uses its zero-based best iteration plus one, otherwise the fitted booster's round count. XGBoost `gblinear` uses the final fitted round count because it does not retain an earlier model. Counts are available with early stopping disabled and with `inline_metrics=False`.

The option is independent of `record_execution` and `post_processing`; leaving it off preserves existing outputs. Python resume must pass the same flag; CLI resume uses the effective setting saved in metadata, including Python overrides. Metadata records an enabled setting, and a changed setting rejects resume before artifacts are rewritten. Older metadata without the flag means `false`. Probability arrays increase JSONL storage with test-window length.

### Recording policy

Use `uel.recording` in YAML or `run(recording={...})` in a persisted Python/MSQ run to omit named groups from `round_data.jsonl.gz`. Every group defaults to `false`; new runs omit bar-level payloads unless explicitly enabled. An explicit all-true policy restores full recording. Values must be literal booleans: unknown keys, parameter references and non-booleans are errors. Explicit policies require both `search_strategy` and `experiment_dir`.

| Key | When `false` |
|---|---|
| `predictions` | Omit saved `preds`, including within retained folds. |
| `alignment` | Omit row-sized `missing_datetimes`, fold `test_datetimes` and contract `signal_rows`; retain dated bounds, source checksums and the full contract digest. |
| `trade_ledger` | Omit the complete event ledger; retain the trade contract, its digest and learning binding. |
| `fold_details` | Omit the entire nested `folds` payload, including fold-local results, bounds and provenance; retain the outer occurrence identity and parameters. |

For example, add this block to an existing YAML `uel` configuration:

```yaml
recording:
  predictions: true
  alignment: true
  trade_ledger: true
  fold_details: true
```

The Python equivalent is `uel.run(..., recording=dict.fromkeys(("predictions", "alignment", "trade_ledger", "fold_details"), True))`. On a new run, `recording=None` uses the configured YAML policy, or all-false defaults when absent. An explicit native mapping replaces the entire YAML policy before unspecified groups default to `false`: native `{"trade_ledger": True}` overrides YAML `{"predictions": True}` with predictions disabled.

For a compiled SFD, its compiled YAML is authoritative even without a separate `yaml_reference`. A conflicting recording declaration in an explicitly supplied `yaml_reference` rejects before artifacts. New persisted recording runs retain the original compiled YAML through the existing `yaml_reference` metadata. The frozen effective policy is stored separately as `metadata.recording`. Changing YAML `uel.recording` changes the existing canonical manifest ID; a native override preserves the source YAML and its ID. Neither operation changes the manifest-ID algorithm. New persisted runs save the effective policy even when no policy was declared.

This policy filters copied serialized records only. Models, predictions used for scoring, scalar CSV/Parquet results, feedback and live `post_processing` retain their full evidence. When folds remain, the other controls apply within them. Walk-forward `trial_returns.parquet` and DSR computation are unchanged. Existing `record_execution` and `record_model_outputs` remain independent and off by default: omitted predictions can coexist with recorded execution, market returns or probabilities.

Disabling `fold_details` or `alignment` makes PBO unavailable because the saved log lacks its required fold/timestamp evidence. These omissions are allowed unless PBO is selected through `acceptance.max_pbo`; that conflict raises before run artifacts are created. Without a PBO threshold, the report records `pbo: null` with an explanation while retaining DSR. Directly requesting `max_pbo` from `acceptance_report` on an incompatible saved log also raises before either report file is written. PBO requires explicitly enabling both `alignment` and `fold_details`; full recording retains existing acceptance behavior.

Default runs and any policy containing `false` are non-resumable, even if that group is absent from its particular records. Python and CLI reject lighter saved/requested policies or changed policies before modifying artifacts. On resume, an omitted policy uses saved metadata; legacy metadata without a policy means all-true recording. Existing full logs retain their resume behavior; enable all four groups when creating a new resumable run.

Supported single-split Trainer/Sensor reconstruction and built-in Cohort selection retain their identities, parameters, scalar results and required exact trade/learning bindings. Omitted row-level evidence cannot be inspected from the log; custom readers must require the fields they use. Existing walk-forward Trainer limitations remain. Reducing recording changes stored evidence, not live retention or model/preparation memory.

The [recorded payload inventory](https://github.com/Vaquum/Limen/blob/main/tests/fixtures/recording_payload_inventory.json) measures two production rounds per case from `spot_1h_20240101_20241231.parquet`, with both execution and model-output recording enabled for ordinary cases and execution recording enabled for walk-forward. For records `R`, logical bytes are `L(R) = sum(len((json.dumps(r) + '\n').encode('utf-8')) for r in R)`, using Python's default JSON settings. Each section's weight is `L(full) - L(without_section)`; its percentage divides that difference by `L(full)` for that case alone. Compressed sizes include the round log and shared market blobs. The alignment measurement removes the entire alignment object; the policy retains its dated bounds, so that percentage is not its omission saving.

| Case | Logical bytes | Compressed bytes | Largest payload sections |
|---|---:|---:|---|
| Rule snapshot | 17,680 | 5,987 | Execution 53.85%; market 37.69% |
| Binary snapshot | 25,376 | 9,790 | Execution 43.08%; market 26.08%; probabilities 22.67% |
| Ridge snapshot | 22,399 | 8,408 | Execution 37.49%; market 29.55%; predictions 27.91% |
| Event | 245,352 | 36,981 | Ledger 89.13%; retained trade contract 10.23% |
| Walk-forward | 31,677 | 10,311 | Fold results 34.99%; alignment 20.34%; execution 19.41% |

In the recorded full-payload baseline, omitting only the event ledger removes 218,682 of the two-event sample's 245,352 logical bytes (89.13%) and recompresses it to 4,668 bytes. With alignment recording disabled, per-signal identities are also omitted; Trainer compares reconstructed contract fields and the original full contract digest. These small fixture observations are not campaign projections or universal savings claims; defaults already omit execution/market and model-output recordings, and compressed omission savings are not additive. Stored counts describe the measured baseline, not a portable exact-byte oracle for other numerical libraries or zlib implementations.

Reproduce the event probe from the repository root using the existing recorded-execution helpers:

```python
import gzip
import json
import tempfile
from pathlib import Path
import polars as pl
from tests import test_record_execution as probe

with tempfile.TemporaryDirectory() as temporary:
    directory = Path(temporary)
    config = probe._config('rule')
    config['sfd']['manifest']['backtest'] = {'product': {
        'kind': 'cash_spot', 'instrument': 'BTCUSDT', 'base_currency': 'BTC',
        'quote_currency': 'USDT', 'quantity_step': 1e-9, 'min_notional': 0,
    }}
    bars = pl.read_parquet('tests/fixtures/spot_1h_20240101_20241231.parquet').head(1200)
    loop = probe._loop(config, bars, directory)
    probe._run(loop, record_execution=True, record_model_outputs=True, recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True))
    full = [(json.dumps(record) + '\n').encode('utf-8') for record in probe._records(directory)]
    lean = [(json.dumps({k: v for k, v in json.loads(row).items() if k != 'trade_ledger'}) + '\n').encode('utf-8') for row in full]
    print('logical full/without ledger:', sum(map(len, full)), sum(map(len, lean)))
    print('compressed full/without ledger:', *(sum(len(gzip.compress(row, compresslevel=1, mtime=0)) for row in rows) for rows in (full, lean)))
```

Structural and relative-size regression checks run with `python -m pytest -q tests/test_recording_policy.py -k payload_projection`; exact compressed counts are not the acceptance criterion.

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
    recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True),
)
```

The budget reducer can trim the remaining queue during a feedback cycle, so the number of completed rows may be lower than the requested permutation budget. `results.csv` and `round_data.jsonl.gz` track completed rounds; `audit.jsonl` records the intervention; checkpoints follow `checkpoint_interval`.

## Resume in practice

Python/MSQ runs can resume when all four recording groups were enabled on the initial run, and an `experiment_dir` and a checkpoint are available:

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

New result directories from 5.31.1 declare `round_storage: gzip-jsonl-v2` in metadata and write `round_data.jsonl.gz`. Each gzip member contains one UTF-8 JSON round plus a newline, compressed at fixed level 1 with mtime zero and no filename header. Retained walk-forward fold objects receive gzip compression only; the [recording policy](#recording-policy) determines their saved fields. Existing directories without the discriminator retain plain `round_data.jsonl`; directories declaring `gzip-jsonl-v1` keep their original compressed representation, including on resume. Neither layout is migrated. Compression preserves the existing recording flags.

For ordinary rounds, a non-null top-level `market` is replaced in place by `market_ref`. Its lowercase SHA-256 digest addresses `round_market/<digest>.json.gz`. The decompressed blob bytes are exactly `json.dumps(market, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('utf-8')`; hash those bytes and verify them before reconstruction. Alignment stays in the owning round. Restore the market key at the reference's position; missing/null market stays missing/null. This preserves exact values, numeric types, signed zero and ordering without lossy conversion. Existing nonfinite tokens elsewhere in round/fold results retain the legacy JSON behavior.

In v2 ordinary rounds, retained top-level `preds` and `probs` flat finite numeric/boolean/null arrays are replaced at their original key position by `preds_ref` or `probs_ref`. Each reference contains `sha256` and the integer element `count`; its digest addresses `round_signals/<digest>.json.gz`. Blob bytes use the same canonical JSON encoding as market blobs. The reader verifies the hash, flat scalar types and declared count before restoring the array; references sharing a round must agree on row count and any retained execution positions. Integer/float/boolean distinctions, nulls and signed zero survive JSON reconstruction. Missing/null fields and nested or nonfinite values retain their inline representation. Nested walk-forward fold payloads are unchanged.

References are local to one run. They share array bytes, while each occurrence keeps its identity, typed parameters, alignment, threshold rule, metrics and execution evidence. All local evidence remains retained and the existing private reader resolves it for Trainer, Cohort, acceptance and resume. The recording policy still controls which groups are saved. There is no additional switch, sink, acknowledgement, receipt or retention policy.

External readers must verify each member trailer before exposing its round. The internal shared reader uses `zlib.decompressobj(wbits=31)`, requires each gzip header, rejects zero padding, and checks EOF/CRC/ISIZE. It validates references before constructing a path and verifies blob digest and return length against predictions, or retained execution positions when predictions are omitted. Decompression is bounded to 512 MiB per round and 128 MiB per blob. The package-local example above uses the private shared reader; this is not a new public reader API.

Trainer, Cohort and acceptance processing reject incomplete artifacts. Live/resume reads may ignore only an incomplete final member outside the successful checkpoint prefix. Resume validates ordered successful CSV identities, scans existing member/line offsets, and truncates in place: committed round-data bytes and inode survive. CSV retains its existing rewrite behavior. This is sequential restoration, not indexed random access.

Blob publication precedes its referencing member through a same-directory temporary file and atomic replace. The contract covers process interruption, not new power-loss/fsync durability or multiple writers in one directory. Orphan blobs and temporary leftovers are tolerated, never used as alternate references, and included in storage accounting; no garbage collector is introduced. Unknown/conflicting layouts and integrity failures raise errors before successful resume or report replacement.

Sharing benefits repeated arrays. Unique or small arrays can cost more, including filesystem allocation and reference overhead; compression and deduplication do not bound total run storage. Neither changes worker IPC, model execution or scientific results. Unique markets still require unique files and save less. Grid currently pins its Limen package version: update its reader before upgrading that separate dependency pin. No historical run is converted automatically.

### Recorded signal-sharing measurement

Reproduce the implemented-format comparison with `python -m pytest -q -s tests/test_round_signals.py -k recorded_storage_savings --log-cli-level=INFO`. It uses the existing recorded hourly fixture, fixed train/validation/test dates, deterministic single-threaded LightGBM parameters and a 16-round factorized sweep across four fee values. Accounting includes the round log, all market/signal blobs and reference metadata; resolved records must equal the v1 representation exactly. The test reports single-round costs as well as the repeated workload. These are small recorded-workload measurements, not campaign projections or a throughput promise.

Measured on 5.31.1 using fixture SHA-256 `a634bbd4` (prefix), the first 1,200 rows and 503 retained test rows. Splits are train Jan 1–20, validation Jan 20–25 and test Jan 25–Feb 15, 2024. The test pins lookback `[2, 4]`, LightGBM estimators `[16, 32]` and fees `[4, 9, 14, 19]`; the remaining deterministic model parameters and recording flags are in its `economic_sweep` fixture. Four distinct prediction/probability pairs recur across the fee values.

| Recorded workload | v1 bytes, all recording files | v2 bytes, all recording files | v1 / v2 allocated bytes |
|---|---:|---:|---:|
| One round | 16,960 | 16,852 | 24,576 / 32,768 |
| Four distinct pairs | 52,388 | 51,966 | 61,440 / 90,112 |
| Sixteen rounds, four repeated pairs | 193,534 | 131,557 | 200,704 / 167,936 |

The repeated case saves 61,977 bytes (32.0%), including format metadata and every market/signal blob. It uses eleven recording files rather than three. The production sweep took 0.354 seconds; the separate serialization samples took 0.016 seconds for v1 and 0.027 seconds for v2. These single local samples show write overhead, not a speedup. Allocated bytes depend on filesystem block size; unique/single cases can use more blocks despite slightly fewer serialized bytes. The regression asserts a relative saving on the repeated recorded case, not fixed compressed sizes or timing.
