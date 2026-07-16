# Experiment Manifest

The Experiment Manifest is Limen's declarative experiment definition. The canonical operator form is YAML: define the manifest, validate it, profile it, run it through CLI, then inspect the generated artifacts.

Use YAML manifests for Limen's opinionated pipeline. Use Python builders, custom `prep()`, and custom `model()` only when the workflow cannot fit the declarative operator path.

The default path provides:

- split-first execution
- train-only fitting for targets and scalers
- automatic data fetching
- cleaner collaboration surfaces
- reproducible experiment definitions and artifact-backed runs

## Prerequisites

- `pip install "vaquum-limen[data]"` for the bundled data-backed templates
- a YAML file created with `limen init` or a concrete Python `MLManifest`/`RuleBasedManifest`
- the optional model extra required by the selected reference architecture

## YAML First Run

```bash
limen init logreg-first.yaml --template logreg_binary
limen validate logreg-first.yaml
limen profile logreg-first.yaml
limen run --dry-run logreg-first.yaml
limen run logreg-first.yaml
```

The run writes a result directory containing the copied YAML manifest, `metadata.json`, `results.csv`, and `round_data.jsonl`.

## YAML Shape

This minimal shape shows the same pieces as the Python builder API:

```yaml
schema_version: "1.0"

metadata:
  name: logreg-first
  limen_version: "5.6.2"
  mode: development

sfd:
  manifest:
    type: ml
    data_source:
      method: limen.data.HistoricalData.get_spot_klines
      params:
        kline_size: 3600
    split_dates:
      train_start: "2024-01-01"
      train_end: "2024-09-30"
      val_start: "2024-10-01"
      val_end: "2024-11-30"
      test_start: "2024-12-01"
      test_end: "2024-12-31"
    indicators:
      - func: limen.indicators.roc
        params:
          period: "{roc_period}"
    target:
      name: quantile_flag
      class: limen.targets.QuantileBinaryTarget
      fit_params:
        source_column: "roc_{roc_period}"
        quantile: "{q}"
      transform_params:
        shift: "{shift}"
    scaler:
      from_params: scaler_type
    reference_architecture: limen.sfd.reference_architecture.logreg_binary
  params:
    roc_period: [1, 4, 12]
    q: [0.35, 0.40, 0.45]
    shift: [-1, -2, -3]
    scaler_type: [logreg]
    C: [0.1, 1.0, 5.0]

uel:
  n_permutations: 25
  search_strategy:
    type: random
  prep_each_round: true
  output_format: csv
```

`schema_version` is the internal Limen YAML contract marker, not a published JSON Schema or editor schema. The current schema version is `1.0`; validation warns when a manifest uses a different string, but compatibility is governed by the Python validator and compiler in this repository.

`uel.n_permutations` is a positive integer execution budget. YAML validation rejects bool, zero, negative, and over-budget values before execution.

### The `uel` Execution Block

The `uel` block configures how the experiment runs. `n_permutations` is required; the rest are optional and fall back to the defaults below.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `n_permutations` | int | required | round budget; rejected if bool, zero, negative, or larger than the parameter space |
| `search_strategy.type` | `random` or `grid` | `random` | `random` lazily samples the parameter space; `grid` enumerates it exhaustively |
| `prep_each_round` | bool | `true` | run prep every round; required for manifest-driven SFDs |
| `checkpoint_interval` | int | `1000` | rounds between checkpoint writes |
| `feedback_interval` | int | `100` | rounds between feedback-controller triggers |
| `pruning_strategies` | list | none | reducers that run during feedback cycles; each item has `type` (a reducer key) and optional `params` |
| `output_format` | `csv` or `parquet` | `csv` | `parquet` also writes `results.parquet`; `results.csv` is always written |
| `output_path` | str | `{name}_{datetime}` | output-directory template; `{name}` and `{datetime}` are substituted, and it is ignored for committed-manifest URI runs |

NOTE: `search_strategy.seed` seeds `random` search for reproducible sampling. `grid` search enumerates the full space and ignores a seed.

`pruning_strategies` declares reducers that fire every `feedback_interval` rounds during a run — see [Reducers And Feedback](Reducers-And-Feedback.md) for the reducer catalog, YAML syntax, and tuning.

## Manifest Types

There are two manifest subclasses under the YAML and Python surfaces:

- **`MLManifest`** — for ML pipelines. Supports indicators, features, targets, scalers, ablation, and calibration.
- **`RuleBasedManifest`** — for rule-based pipelines. Supports indicators and features, plus `with_strategy()` for predicate-driven entry signals. Does not support scalers or ablation.

Both subclasses share a common base (`Manifest`) that owns data fetching, splitting, and model execution. The `manifest()` function in every foundational SFD returns the base `Manifest` type as a uniform interface, while internally constructing the correct subclass. `Manifest` itself is not meant to be instantiated directly — its `prepare_data()` raises `NotImplementedError`.

## Python Builder Equivalent

The Python builder API is the equivalent surface for foundational SFD authors and custom integrations.

```python
from limen.data import HistoricalData
from limen.experiment import Manifest
from limen.experiment import MLManifest
from limen.features import kline_imbalance, vwap
from limen.indicators import atr, ppo, roc, wilder_rsi
from limen.scalers import LogRegScaler
from limen.sfd.reference_architecture import logreg_binary
from limen.targets import QuantileBinaryTarget

def params():
    return {
        'shift': [-1, -2, -3],
        'q': [0.35, 0.40, 0.45],
        'roc_period': [1, 4, 12, 24],
        'class_weight': [0.45, 0.65, 0.85],
        'C': [0.1, 1.0, 5.0],
        'max_iter': [60, 120, 180],
        'solver': ['lbfgs', 'newton-cg'],
        'tol': [0.001, 0.01],
    }

def manifest() -> Manifest:
    return (
        MLManifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'start_date_limit': '2025-01-01'},
        )
        .set_test_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 7200, 'row_count_limit': 5000},
        )
        .set_split_config(8, 1, 2)
        .add_indicator(roc, period='roc_period')
        .add_indicator(atr, period=14)
        .add_indicator(ppo)
        .add_indicator(wilder_rsi)
        .add_feature(vwap)
        .add_feature(kline_imbalance)
        .with_target_label(
            'quantile_flag',
            QuantileBinaryTarget,
            fit_params={'source_column': 'roc_{roc_period}', 'quantile': 'q'},
            transform_params={'shift': 'shift'},
        )
        .set_scaler(LogRegScaler)
        .with_reference_architecture(logreg_binary)
    )
```

Run the Python-built SFD through UEL only when direct engine integration is required:

```python
import limen
from limen import sfd

uel = limen.UniversalExperimentLoop(sfd=sfd.foundational_sfd.logreg_binary)

uel.run(
    experiment_name='manifest-demo',
    n_permutations=25,
    prep_each_round=True,
)
```

For ordinary runs, keep the manifest in YAML and use `limen run`.

## How A Manifest Executes

The manifest pipeline is split-first by design. In other words, fitting never happens on the full dataset before the train, validation, and test partitions are created.

The execution order is:

1. fetch raw input data
2. optionally apply a pre-split selector
3. split into train, validation, and test
4. optionally form bars inside each split
5. apply indicators and features
6. compute fitted target parameters on train, then apply target transforms across splits
7. optionally perform feature ablation
8. fit the scaler on train, then apply it across splits
9. finalize a standard `data_dict`
10. run the configured model with round-specific parameters

This ordering is one of the main reasons to use a manifest: the leak-prevention rules come built in.

The manifest is deliberately opinionated. It forces split-first execution, train-only fitting, and immutable dataframe-style transforms because those guardrails prevent leakage-prone financial-ML mistakes.

There is one more protection step after feature engineering: non-empty splits are aligned to a shared column set before the final `data_dict` is built. If a transform such as `fractional_diff` produces a column in one split but not another because the shorter split lacks enough history, Limen drops that extra column from the non-empty splits so the downstream model still receives a consistent schema.

## Data Source Configuration

### `set_data_source(method, params=None)`

Configure the production data source for the manifest.

```python-fragment
from limen.data import HistoricalData

.set_data_source(
    method=HistoricalData.get_spot_klines,
    params={'kline_size': 3600, 'start_date_limit': '2025-01-01'},
)
```

### `set_test_data_source(method, params=None)`

Configure the test data source used when `test_mode=True` is passed to `UniversalExperimentLoop`.

```python-fragment
.set_test_data_source(
    method=HistoricalData.get_spot_klines,
    params={'kline_size': 7200, 'row_count_limit': 5000},
)
```

### Data source selection

Pass `test_mode=True` to `UniversalExperimentLoop` to fetch from the test data source instead of the production source.
When `test_mode=True` and a test data source is configured, `fetch_test_data()` is called; otherwise `fetch_data()` is used.

That is why foundational SFDs can run locally with no explicit `data=` and still use the configured test data source.
To keep local runs bounded, configure `set_test_data_source()` with explicit `kline_size` and `row_count_limit`.

## Pipeline Configuration

### `set_split_config(train, val, test)`

Configure the relative split sizes.

```python-fragment
.set_split_config(8, 1, 2)
```

This means 8/11 train, 1/11 validation, and 2/11 test.

Behavior rules:

- `train` must be positive
- `val` and `test` can be zero but not negative
- the method raises `ValueError` if those constraints are violated

Allowing zeros supports explicit workflows where validation or test data is intentionally absent. For example, a caller-created clone with `split_config=(1, 0, 0)` fits on all available rows. `Trainer` does not apply that override: it replays the original split so it can compare the original metrics.

### `set_split_dates(train_start, train_end, val_start, val_end, test_start, test_end, *, val_predict_guard=True, test_predict_guard=True)`

Pin the train, val, and test windows to absolute `datetime` bounds, in preference to ratio-based splits whose absolute boundary depends on the input row count.

```python-fragment
from datetime import datetime

.set_split_dates(
    datetime(2024, 1, 1), datetime(2024, 7, 1),
    datetime(2024, 7, 1), datetime(2024, 10, 1),
    datetime(2024, 10, 1), datetime(2025, 1, 1),
)
```

Behavior rules:

- Windows are half-open `[start, end)`. A row enters a split iff its `datetime` falls inside that window.
- Ordering must satisfy `train_start <= train_end <= val_start <= val_end <= test_start <= test_end`. Gaps between adjacent windows are allowed; rows inside a gap are intentionally excluded from all three splits.
- Every bound must be a `date` or `datetime` instance. Strings, ints, and floats raise `TypeError` at the API boundary.
- `val_predict_guard` and `test_predict_guard` are keyword-only and default `True`. They control whether the served `Sensor` masks predictions inside the val and test windows. Both `True` masks the full `[train_start, test_end)` envelope — every served prediction is identical to omitting them; setting one to `False` makes that window emit real `0/1` predictions instead of `None`, so a served cohort can be checked against its own test-split backtest. Train is always masked (the model trains on it) and has no flag. Each flag must be a `bool` or `set_split_dates` raises `TypeError`.
- When `split_dates` is set it takes precedence over `set_split_config` in both `prepare_data` and `compute_test_bars`. The split runs on the raw data (before per-split feature transforms), so transforms can still drop rows inside a slice but cannot move rows across slice boundaries.
- `with_params_override(split_config=(1, 0, 0))` clears any previously pinned `split_dates`, so an explicit ratio override is not silently shadowed by an earlier date pin.

Use this when the train / val / test boundaries must land on specific datetimes (e.g. honoring a deployment date), and `set_split_config` when proportions of the row count are the natural way to express the split.

In a YAML manifest the same windows and flags live in the `split_dates` block. `val_predict_guard` and `test_predict_guard` are optional and default to `true`; a `train_predict_guard` key (or a non-boolean value for either flag) fails `limen validate`:

```yaml
split_dates:
  train_start: "2024-01-01"
  train_end:   "2024-07-01"
  val_start:   "2024-07-01"
  val_end:     "2024-10-01"
  test_start:  "2024-10-01"
  test_end:    "2025-01-01"
  val_predict_guard:  false   # val window emits real predictions
  test_predict_guard: false   # test window emits real predictions
```

### `set_pre_split_data_selector(func, **params)`

Optionally select or reduce the raw dataset before splitting.

```python-fragment
from limen.data.utils import random_slice

.set_pre_split_data_selector(
    random_slice,
    rows='random_slice_size',
    safe_range_low='random_slice_min_pct',
    safe_range_high='random_slice_max_pct',
    seed='random_seed',
)
```

Use this for smaller or controlled slices of the raw dataset before the normal split-first pipeline begins.

### `set_bar_formation(func, **params)`

Configure threshold-bar formation inside each split.

```python-fragment
from limen.data.utils import compute_data_bars

.set_bar_formation(
    compute_data_bars,
    bar_type='bar_type',
    volume_threshold='volume_threshold',
)
```

See [Data Bars](Data-Bars.md) for the supported bar types and output schema.

### `set_required_bar_columns(columns)`

Assert that bar formation still leaves the downstream columns required by the experiment.

```python-fragment
.set_required_bar_columns([
    'datetime',
    'open',
    'high',
    'low',
    'close',
    'volume',
    'no_of_trades',
])
```

This assertion protects model or backtest paths that require OHLC fields after bar formation.

## Indicators And Features

`add_indicator()` and `add_feature()` use the same underlying mechanism. Both add transformation steps to the split-first pipeline.

### `add_indicator(func, group=None, include_if=None, **params)`

```python-fragment
.add_indicator(roc, period='roc_period')
.add_indicator(wilder_rsi)
```

### `add_feature(func, group=None, include_if=None, **params)`

```python-fragment
.add_feature(vwap)
.add_feature(kline_imbalance)
```

### Parameter resolution

Manifest parameters are resolved at run time:

- literal scalars are passed through unchanged
- a string matching a `round_params` key is looked up from the current round
- a formatted string like `'roc_{roc_period}'` is interpolated from `round_params`
- strings starting with `_` are treated as fitted-parameter references when available

That means this:

```python-fragment
.add_indicator(roc, period='roc_period')
```

becomes:

```python
roc(data, period=round_params['roc_period'])
```

for each round.

### Group filtering

Use `group=` to tag transforms into families, then filter by `feature_groups` in `round_params`.

```python-fragment
.add_indicator(roc, group='momentum', period='roc_period')
.add_indicator(wilder_rsi, group='momentum')
.add_feature(vwap, group='microstructure')
```

If `round_params['feature_groups']` is present, only transforms whose group is in that string are applied. Ungrouped transforms always run. The value is a pipe-delimited string of active group names, or `'all'` to include every group.

### Conditional inclusion

Use `include_if=` when a transform should only run if a boolean round parameter is true. Missing control keys are treated as false.

```python-fragment
.add_feature(vwap, include_if='use_vwap')
```

## Parameter-Controlled Perturbations

The manifest builder supports perturbation-style workflows directly in the declarative surface.

### Feature-group selection

Use `group=` on indicators or features, then pass `feature_groups` in `round_params`.

```python-fragment
.add_indicator(roc, group='momentum', period='roc_period')
.add_feature(vwap, group='microstructure')
```

At run time:

```python
round_params = {'feature_groups': 'momentum'}
```

Use `|` to activate multiple groups: `'momentum|microstructure'`. Use `'all'` to include every group.

Only grouped transforms in the selected families run. Ungrouped transforms always run.

### Conditional feature toggles

Use `include_if=` for boolean on/off switches:

```python-fragment
.add_feature(vwap, include_if='use_vwap')
```

At run time:

```python
round_params = {'use_vwap': False}
```

The transform is skipped.

### Feature ablation

Use `set_feature_ablation()` to let the manifest randomly drop feature columns after transforms and before scaling.

```python
manifest.set_feature_ablation()
```

Then control it from round parameters:

```python
round_params = {
    'feature_drop_count': 1,
    'feature_drop_seed': 42,
}
```

Important behavior:

- the manifest mutates `round_params` by adding `_dropped_features`
- protected columns such as `datetime` and the target are not eligible
- the same seed reproduces the same dropped columns

The selected column depends on the eligible feature set. The exact list is recorded in `round_params['_dropped_features']`.

In artifact-backed runs, that `_dropped_features` payload is stored into `round_data.jsonl`, and [Trainer](Trainer.md) reproduces the same drop set during promotion.

### Scaler choice from parameters

Use `set_scaler_from_params()` when scaler selection itself is part of the search space.

```python
manifest.set_scaler_from_params('scaler_type')

params = {
    'scaler_type': ['logreg', 'robust', 'rank_gauss'],
}
```

The registry resolves:

- `RobustScaler` when `scaler_type='robust'`
- `RankGaussScaler` when `scaler_type='rank_gauss'`

### Manifest-level overrides

Use `with_params_override(split_config=(1, 0, 0))` for a manifest clone with structural changes that should not come from the round search itself.

Examples:

- `split_config=(1, 0, 0)` for an explicit caller-owned all-data fit
- `start_date_limit='2025-01-01'` for a controlled data-window variant
- `row_count_limit=5000` for test or smoke paths

Round parameters handle search-time variation. `with_params_override(split_config=(1, 0, 0))` handles external structural control.

Then in `params()`:

```python
{
    'use_vwap': [True, False],
}
```

## Target Configuration

Target construction uses `Manifest.with_target_label()` with a class from `limen.targets`. The class is fitted once on the training split and then applied to validation and test without refitting.

```python-fragment
from limen.targets import QuantileBinaryTarget

.with_target_label(
    'quantile_flag',
    QuantileBinaryTarget,
    fit_params={'source_column': 'roc_{roc_period}', 'quantile': 'q'},
    transform_params={'shift': 'shift'},
)
```

`fit_params` are forwarded to `__init__` on the training split. `transform_params` are forwarded to `transform()` on every split. Both support round-param references (`'q'` resolves to the current round's `q` value).

The target column is placed last before Limen finalizes the `data_dict`.

See [Targets](Targets.md) for the full reference including all built-in target classes and the custom target class convention.

## Scaling

### `set_scaler(transform_class, param_name='_scaler', extra_params=None)`

Configure a fitted scaler that is instantiated on train and then applied across the splits.

```python-fragment
from limen.scalers import LogRegScaler

.set_scaler(LogRegScaler)
```

The fitted scaler is stored in the resulting `data_dict` under `_scaler`.

Pass `extra_params` to forward constructor arguments. String values are treated as sweep param references resolved from `round_params` at fit time; all other values are static:

Underscore-prefixed strings, such as `_scaler`, resolve to fitted parameters only when that key is available in the current fitted-parameter map. Otherwise they remain literal strings.

```python-fragment
# window swept as a param; min_samples fixed
.set_scaler(CausalRollingRobustScaler, extra_params={'window': 'scaler_window', 'min_samples': 25})
```

### `set_scaler_from_params(param_name='scaler_type', extra_params=None)`

Select a scaler dynamically from `SCALER_REGISTRY` using a round parameter.

```python-fragment
.set_scaler_from_params('scaler_type')
```

Then in `params()`:

```python
{
    'scaler_type': ['logreg', 'linear', 'robust'],
}
```

Use this when scaler choice is itself part of the search space. Pass `extra_params` to forward constructor arguments using the same static/dynamic resolution as `set_scaler()`.

### `set_strict_mode(strict_mode)`

Enable strict null checking after the Context Carry-Over (CCO) pipeline.

```python-fragment
.set_strict_mode(True)
```

#### Background: Context Carry-Over

When `prepare_data` processes val and test splits, it prepends a raw tail of the preceding split (the CCO block) before running feature transforms and the scaler. This ensures indicator warm-up rows and rolling-scaler warm-up rows are fully warmed before the first real split row is produced — matching how the Sensor sees a continuous stream in live inference.

After the CCO rows are stripped from each processed split, two null checkpoints run on feature columns only (the target column, which always has a trailing null from `shift:-1`, is excluded):

- **Checkpoint A** — after leading warm-up nulls are sliced off, before the scaler. Nulls here indicate mid-split data gaps in the raw input or indicator failures.
- **Checkpoint B** — after the scaler and after CCO rows are removed. Nulls here would indicate the scaler itself introduced them (rare).

Checkpoint A also runs on the training split, where there is no CCO block.

#### Strict vs non-strict

`strict_mode=True` — any unexpected null raises `StrictModeError` with the split name, checkpoint label, column name, and the timestamps of the null rows. The UEL catches this per-permutation, records the error message in `results.csv` under the `strict_mode_error` column (metric columns are empty for that round), and continues to the next round.

`strict_mode=False` (default) — the same detection runs, but logs a `WARNING` instead of raising. Training continues and the final `drop_nulls()` silently removes the affected rows.

All foundational ML SFDs and YAML templates ship with `strict_mode=True`. New SFDs should follow the same convention — silent drops during search can mask data quality problems that only appear under specific scaler or param combinations.

## PCA Compression

### `set_pca_compression(enabled_param='auto_pca', n_components_param='pca_k', scaler_param_name='_scaler', component_prefix='pc_')`

Configure optional PCA feature compression for ML manifests.

```python
from limen.scalers import RobustScaler

manifest = (
    MLManifest()
    .set_scaler(RobustScaler)
    .set_pca_compression()
)
```

When `round_params[enabled_param]` is absent or `False`, the manifest uses the
current scaled feature surface unchanged.

When it is `True`, `round_params[n_components_param]` must be an integer. The
manifest fits full-SVD PCA on the train split only, then applies that frozen
rotation to validation and test. `x_train`, `x_val`, and `x_test` are replaced
with `pc_*` columns; no parallel raw-feature dataset is retained.

PCA compression requires the configured fitted scaler to be `RobustScaler`.
The PCA input is assumed to be stationary upstream. Limen does not run
stationarity shims or tests in this path.

## Rule-Based Strategy

Use `with_strategy()` when the strategy is expressed as boolean predicate logic over indicator columns — no Python model code required. This path is for rule-based SFDs only and produces a different `data_dict` shape than the ML path.

### `with_strategy(conditions, entry)`

Configure a rule-based strategy from a list of condition config dicts and an entry signal id.

```python
from limen.experiment import RuleBasedManifest
from limen.sfd.reference_architecture.rule_based import rule_based

conditions = [
    {'id': 'rsi_oversold', 'type': 'threshold', 'column': 'wilder_rsi_{rsi_period}', 'operator': '<', 'value': '{rsi_threshold}'},
    {'id': 'above_ema',    'type': 'relative',   'column': 'close', 'operator': '>', 'other_column': 'ema_{ema_period}'},
    {'id': 'entry',        'operator': 'and',     'operands': ['rsi_oversold', 'above_ema']},
]

manifest = (
    RuleBasedManifest()
    .add_indicator(wilder_rsi, period='rsi_period')
    .add_indicator(ema, period='ema_period')
    .with_strategy(conditions, entry='entry')
    .with_reference_architecture(rule_based)
)
```

`with_strategy()` validates the config at construction time — it raises `ValueError` if any condition is missing an `id`, if ids are not unique, if `entry` does not refer to a known id, or if compound conditions reference unknown operand ids.

### Condition schema

Each condition is a dict with an `id` field. There are two kinds:

**Leaf conditions** — evaluate directly against data columns. The `type` field determines which predicate function is used:

| type | required fields | description |
|------|----------------|-------------|
| `threshold` | `column`, `operator`, `value` | column compared against a constant |
| `relative` | `column`, `operator`, `other_column` | column compared against another column |
| `crossover` | `column`, `other_column`, `direction` (`above`/`below`) | cross event, True on the bar of the cross |
| `slope` | `column`, `direction` (`rising`/`falling`), `lookback` | column rising or falling over a lookback window |
| `sql_expr` | `expr` | SQL expression string — column names referenced directly, no `pl.col()` wrapper needed |

**Compound conditions** — combine leaf or other compound conditions using boolean logic:

```python
{'id': 'entry', 'operator': 'and', 'operands': ['rsi_oversold', 'above_ema']}
{'id': 'exit',  'operator': 'or',  'operands': ['condition_a', 'condition_b']}
{'id': 'not_high_vol', 'operator': 'not', 'operands': ['high_vol']}
```

Supported operators: `and`, `or`, `not`. The `not` operator requires exactly one operand.

### Temporal modifiers

Any leaf condition can be wrapped with a temporal modifier by adding an optional field:

```python
{'id': 'rsi_oversold', 'type': 'threshold', 'column': 'wilder_rsi_14', 'operator': '<', 'value': 30, 'persistence_n': 3}
# True only if RSI has been below 30 for 3 consecutive bars

{'id': 'macd_cross', 'type': 'crossover', 'column': 'macd', 'other_column': 'signal', 'recency_n': 5}
# True if the crossover happened within the last 5 bars
```

`persistence_n` and `recency_n` are mutually exclusive — only one can be specified per condition.

### Parameter template substitution

Column names and values can reference round parameters using `{param}` placeholders:

```python
{'id': 'rsi_oversold', 'type': 'threshold', 'column': 'wilder_rsi_{rsi_period}', 'operator': '<', 'value': '{rsi_threshold}'}
```

These are resolved per round from `round_params`. The corresponding indicator must be added to the manifest so the column is present in the data.

### Direct position transforms

A rule-based transform can emit its complete long/flat position directly. The strategy then needs only one threshold leaf over that output:

```yaml
indicators:
  - func: limen.features.dollar_bar_crash_reversal
    params:
      momentum_threshold_bps: "{momentum_threshold_bps}"
      flow_z_threshold: "{flow_z_threshold}"
      hold_minutes: "{hold_minutes}"
strategy:
  conditions:
    - id: crash_reversal_position
      type: threshold
      column: dollar_bar_crash_reversal_position
      operator: ">"
      value: 0
  entry: crash_reversal_position
backtest:
  fee_bps: 10.0
  slip_bps: 5.0
```

This is the shape used by the bundled `dollar_bar_crash_reversal` template. `RuleBasedStrategy` evaluates the position with the declared costs and reports `pnl_per_trade_bps_{split}` with its `num_executed_trades_{split}` denominator in addition to the generic snapshot metrics.

### Restrictions

Rule-based manifests cannot use scalers or feature ablation. Both will raise `ValueError` at data-preparation time:

- `set_scaler()` / `set_scaler_from_params()` — predicates depend on original indicator scales; scaling RSI to `[0,1]` makes threshold comparisons like `< 30` meaningless.
- `set_feature_ablation()` — predicate columns are derived from specific indicator columns and cannot be randomly dropped.

### What `prepare_data()` produces for rule-based

The rule-based path produces a different `data_dict` than the ML path:

```python
{
    'train': pl.DataFrame,   # full indicator DataFrame, no x/y split
    'val':   pl.DataFrame,
    'test':  pl.DataFrame,
    'strategy': {
        'conditions': [
            {'id': 'entry', 'type': 'threshold', 'column': 'wilder_rsi_14', 'operator': '<', 'value': 30},
        ],
        'entry': 'entry',
    },
    '_alignment': {
        'first_test_datetime': None,
        'last_test_datetime': None,
        'missing_datetimes': [],
    },
}
```

Predicate boolean columns are added to each split DataFrame before the dict is assembled. The model receives the full DataFrames and applies the boolean logic tree at evaluation time.

### `sql_expr` escape hatch

The `sql_expr` predicate type parses a SQL expression string via `pl.sql_expr()`. Column names are referenced directly without wrappers:

```python
{'id': 'volume_spike', 'type': 'sql_expr', 'expr': 'volume > avg_volume_20 * {multiplier}'}
{'id': 'ratio_check',  'type': 'sql_expr', 'expr': '(close - sma_200) / sma_200 > {pct_threshold}'}
```

This is the safe alternative to writing Python code — the polars SQL parser rejects IO operations and arbitrary function calls by design. Parameter substitution works identically to other predicate types using `{param}` placeholders.

## Model Configuration

### `with_reference_architecture(architecture_function)`

Configure the final model function.

```python-fragment
from limen.sfd.reference_architecture import logreg_binary

.with_reference_architecture(logreg_binary)
```

The model function must accept the prepared `data_dict` as its first argument.

All remaining parameters are auto-mapped from `round_params` by signature inspection. For a model function with this signature:

```python
from limen.sfd.reference_architecture.logreg_binary import LogRegBinary

def logreg_binary(data, C=1.0, class_weight=None, max_iter=100, solver='lbfgs'):
    model = LogRegBinary().train(
        data,
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        solver=solver,
    )
    return model.evaluate(data, inline_metrics=True)
```

then manifest execution will automatically pull `C`, `class_weight`, `max_iter`, and `solver` from the current round when those keys exist in `round_params`.

The built-in `logreg_binary` SFD uses this same path for the full sklearn logistic-regression constructor surface, so model tuning stays in `params()` rather than hidden inside the architecture.

Required model parameters with no defaults must be present in the round params or Limen will raise.

## Calibration

### `with_calibration()`

Opens a `CalibrationBuilder` for configuring probability calibration and threshold optimization. Call `.done()` to finalize and return to the manifest.

```python-fragment
from limen.calibration import grid_threshold_optimizer, sklearn_probability_calibrator
from limen.metrics.balanced_metric import balanced_metric

.with_reference_architecture(logreg_binary)
.with_calibration()
.probability_calibration(func=sklearn_probability_calibrator, method='isotonic')
.threshold_function(func=grid_threshold_optimizer, metric=balanced_metric)
.done()
```

### `CalibrationBuilder.probability_calibration(func, **params)`

Configure the probability calibration step. `func` receives `(clf, x_val, y_val, **params)` and must return a fitted object with `predict_proba()`. Extra keyword arguments are forwarded to `func`; string values matching `round_params` keys are substituted at runtime.

### `CalibrationBuilder.threshold_function(func, **params)`

Configure the threshold optimization step. `func` receives `(y_val, val_proba, **params)` and must return `(threshold, score)`. Extra keyword arguments are forwarded the same way.

### `CalibrationBuilder.done()`

Finalises the config and returns the manifest. Raises `ValueError` if neither `probability_calibration()` nor `threshold_function()` was called first.

### Four modes

The two optional steps give a 2×2 grid of calibration modes, controlled by per-round flags:

| `use_calibration` | `use_threshold` | Behavior |
|---|---|---|
| True | True | calibrate → optimize threshold |
| True | False | calibrate → decide at 0.5 |
| False | True | raw probabilities → optimize threshold |
| False | False | no calibration injected |

Both flags default to `True`. Add `'use_calibration': [True, False]` and `'use_threshold': [True, False]` to `params()` to compare all four modes within a single experiment.

### String param resolution

Calibration params support the same resolution convention as the rest of the manifest. String values are looked up from `round_params` at runtime; non-string values pass through unchanged:

```python-fragment
.probability_calibration(func=sklearn_probability_calibrator, method='cal_method')
.threshold_function(func=grid_threshold_optimizer,
                    metric=balanced_metric,          # callable — passes through unchanged
                    threshold_min='threshold_min')   # string → round_params lookup
```

See [Calibration](Calibration.md) for the full reference including custom calibrators and threshold optimizers.

## Feature Ablation

### `set_feature_ablation(drop_count_key='feature_drop_count', seed_key='feature_drop_seed')`

Configure deterministic drop-N feature ablation after target construction.

```python-fragment
.set_feature_ablation()
```

Then in `params()`:

```python
{
    'feature_drop_count': [0, 1, 2],
    'feature_drop_seed': [42],
}
```

This makes feature robustness part of the search itself.

## Data Dict Extension

### `add_to_data_dict(func)`

Append custom entries to the finalized `data_dict`.

```python-fragment
def extend_data_dict(data_dict, split_data, round_params, fitted_params):
    data_dict['train_rows'] = split_data[0].height
    return data_dict

.add_to_data_dict(extend_data_dict)
```

Use this when the model needs additional structured inputs that are not part of Limen's standard `x_*` / `y_*` schema.

## Manifest Overrides

### `with_params_override(**overrides)`

Create a deep-copied manifest with selected overrides.

```python
manifest_full = manifest.with_params_override(split_config=(1, 0, 0))
```

This supports caller-owned variants, including an explicit change from train/validation/test mode to all-data training. It is not part of `Trainer.train()`, which preserves the original manifest configuration for metric reconstruction.

Supported override behavior today:

- `split_config=(1, 0, 0)` overrides the split ratios
- other keys are interpreted as data-source parameter overrides and are validated against the configured data-source method signature

Example:

```python
manifest_small = manifest.with_params_override(row_count_limit=1000)
manifest_full = manifest.with_params_override(split_config=(1, 0, 0))
```

The original manifest remains unchanged.

## What `prepare_data()` Produces

The manifest ultimately builds Limen's standard `data_dict`.

The foundational logistic-regression manifest produces:

- `x_train`, `y_train`
- `x_val`, `y_val`
- `x_test`, `y_test`
- `price_data_for_backtest`
- `_alignment`
- `_feature_names`
- fitted objects such as `_scaler` and fitted target parameters

`price_data_for_backtest` contains the aligned test-window OHLC data used by Limen's backtest layer.

## Function Contracts

### Indicator and feature functions

Indicator and feature functions are applied during the feature stage over a Polars lazy pipeline.

The practical contract is:

- input: frame with the columns the transform expects
- output: frame with the new columns added
- behavior: deterministic with respect to the resolved parameters

Custom manifest transforms should match Limen's built-ins: accept a frame plus keyword arguments, and return the transformed frame.

### Fitted parameter computation functions

A fitted-parameter compute function is called only on the training split.

The contract is:

- input: eager `pl.DataFrame` plus resolved kwargs
- output: one fitted value

That fitted value is then stored under the fitted-parameter name and can be referenced later in the target or scaler step.

### Target transform functions

Target transforms operate on eager `pl.DataFrame` objects after the feature stage.

The contract is:

- input: eager `pl.DataFrame` plus resolved kwargs
- output: transformed `pl.DataFrame`

Use fitted transforms when the function depends on train-only fitted state. Use plain transforms otherwise.

### Model functions

The architecture function configured in `with_reference_architecture(logreg_binary)` must:

- accept the finalized `data_dict` as its first argument
- accept any searched model parameters as named kwargs
- return a results dictionary compatible with Limen metrics and logging

If the model returns `_preds`, UEL stores them in `uel.preds` and the `Log` layer can use them for post-run analysis.

## Common Recipes

### Binary target from a fitted quantile cutoff

```python-fragment
from limen.targets import QuantileBinaryTarget

.with_target_label(
    'quantile_flag',
    QuantileBinaryTarget,
    fit_params={'source_column': 'roc_{roc_period}', 'quantile': 'q'},
    transform_params={'shift': 'shift'},
)
```

### Search over feature families

```python
def params():
    return {
        'feature_groups': ['momentum', 'momentum|microstructure'],
        'roc_period': [4, 8, 12],
    }

manifest = (
    MLManifest()
    .add_indicator(roc, group='momentum', period='roc_period')
    .add_feature(vwap, group='microstructure')
)
```

### Search over scaler choice

```python
def params():
    return {
        'scaler_type': ['logreg', 'robust'],
    }

manifest = MLManifest().set_scaler_from_params('scaler_type')
```

### Build an explicit all-data variant

```python
manifest_full = manifest.with_params_override(split_config=(1, 0, 0))
```

This is a separate workflow from [Trainer](Trainer.md). Trainer preserves the original split to reproduce logged metrics.

## Read Next

- Continue to [Universal Experiment Loop](Universal-Experiment-Loop.md) to run a manifest-driven SFD.
- Continue to [Calibration](Calibration.md) for the full calibration reference including custom calibrators, threshold optimizers, and all four calibration modes.
- Continue to [Historical Data](Historical-Data.md) for data surfaces a manifest can reference.
- Continue to [Data Bars](Data-Bars.md) if bar formation is part of the experiment design.
- Use [Indicators](Indicators.md), [Features](Features.md), [Targets](Targets.md), [Transforms](Transforms.md), and [Scalers](Scalers.md) as the reference layer while authoring manifests.
