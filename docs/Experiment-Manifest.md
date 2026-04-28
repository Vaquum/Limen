# Experiment Manifest

The Experiment Manifest is Limen's declarative pipeline builder. Instead of hand-writing `prep()` and threading every step yourself, you describe how data should be fetched, split, transformed, targeted, scaled, and handed to a model.

This is the default path for Limen work because it gives you:

- split-first execution
- train-only fitting for targets and scalers
- automatic data fetching
- cleaner collaboration surfaces
- reproducible experiment definitions

Use a manifest when you want Limen's opinionated pipeline. Skip it only when the workflow genuinely needs fully custom `prep()` and `model()` functions.

## Golden Path Example

This is a complete manifest-driven SFD in the style Limen uses today.

```python
from limen.data import HistoricalData
from limen.experiment import Manifest
from limen.features import compute_quantile_cutoff, quantile_flag
from limen.indicators import atr, ppo, roc, wilder_rsi
from limen.features import kline_imbalance, vwap
from limen.scalers import LogRegScaler
from limen.sfd.reference_architecture import logreg_binary
from limen.transforms import shift_column_transform

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

def manifest():
    return (
        Manifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'start_date_limit': '2025-01-01'},
        )
        .set_test_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 7200, 'n_rows': 5000},
        )
        .set_split_config(8, 1, 2)
        .add_indicator(roc, period='roc_period')
        .add_indicator(atr, period=14)
        .add_indicator(ppo)
        .add_indicator(wilder_rsi)
        .add_feature(vwap)
        .add_feature(kline_imbalance)
        .with_target_label('quantile_flag')
            .add_fitted_transform(quantile_flag)
                .fit_param('_quantile_cutoff', compute_quantile_cutoff, col='roc_{roc_period}', q='q')
                .with_params(col='roc_{roc_period}', cutoff='_quantile_cutoff')
            .add_transform(shift_column_transform, shift='shift', column='target_column')
            .done()
        .set_scaler(LogRegScaler)
        .with_reference_architecture(logreg_binary)
    )
```

Run it through UEL like this:

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

The manifest is deliberately opinionated. It forces split-first execution, train-only fitting, and immutable dataframe-style transforms because those guardrails prevent the most common financial-ML mistakes.

There is one more protection step after feature engineering: non-empty splits are aligned to a shared column set before the final `data_dict` is built. If a transform such as `fractional_diff` produces a column in one split but not another because the shorter split lacks enough history, Limen drops that extra column from the non-empty splits so the downstream model still receives a consistent schema.

## Data Source Configuration

### `set_data_source(method, params=None)`

Configure the production data source for the manifest.

```python
from limen.data import HistoricalData

.set_data_source(
    method=HistoricalData.get_spot_klines,
    params={'kline_size': 3600, 'start_date_limit': '2025-01-01'},
)
```

### `set_test_data_source(method, params=None)`

Configure the test data source used when `test_mode=True` is passed to `UniversalExperimentLoop`.

```python
.set_test_data_source(
    method=HistoricalData.get_spot_klines,
    params={'kline_size': 7200, 'n_rows': 5000},
)
```

### Data source selection

Pass `test_mode=True` to `UniversalExperimentLoop` to fetch from the test data source instead of the production source.
When `test_mode=True` and a test data source is configured, `fetch_test_data()` is called; otherwise `fetch_data()` is used.

That is why foundational SFDs can run locally with no explicit `data=` and still use the configured test data source.
To keep local runs lightweight, configure `set_test_data_source()` with explicit `kline_size` and `n_rows`.

## Pipeline Configuration

### `set_split_config(train, val, test)`

Configure the relative split sizes.

```python
.set_split_config(8, 1, 2)
```

This means 8/11 train, 1/11 validation, and 2/11 test.

Behavior rules:

- `train` must be positive
- `val` and `test` can be zero but not negative
- the method raises `ValueError` if those constraints are violated

Allowing zeros is important for retraining workflows such as Trainer Pass 2, where `split_config=(1, 0, 0)` means "fit on all available data."

### `set_pre_split_data_selector(func, **params)`

Optionally select or reduce the raw dataset before splitting.

```python
from limen.data.utils import random_slice

.set_pre_split_data_selector(
    random_slice,
    rows='random_slice_size',
    safe_range_low='random_slice_min_pct',
    safe_range_high='random_slice_max_pct',
    seed='random_seed',
)
```

Use this when you want smaller or controlled slices of the raw dataset before the normal split-first pipeline begins.

### `set_bar_formation(func, **params)`

Configure threshold-bar formation inside each split.

```python
from limen.data.utils import compute_data_bars

.set_bar_formation(
    compute_data_bars,
    bar_type='bar_type',
    volume_threshold='volume_threshold',
)
```

See [Data Bars](Data-Bars.md) for the supported bar types and output schema.

### `set_required_bar_columns(columns)`

Assert that bar formation still leaves the downstream columns your experiment requires.

```python
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

This is especially useful when your model or backtest assumes OHLC fields are present after bars have been formed.

## Indicators And Features

`add_indicator()` and `add_feature()` use the same underlying mechanism. Both add transformation steps to the split-first pipeline.

### `add_indicator(func, group=None, include_if=None, **params)`

```python
.add_indicator(roc, period='roc_period')
.add_indicator(wilder_rsi)
```

### `add_feature(func, group=None, include_if=None, **params)`

```python
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

```python
.add_indicator(roc, period='roc_period')
```

becomes:

```python
roc(data, period=round_params['roc_period'])
```

for each round.

### Group filtering

Use `group=` to tag transforms into families, then filter by `feature_groups` in `round_params`.

```python
.add_indicator(roc, group='momentum', period='roc_period')
.add_indicator(wilder_rsi, group='momentum')
.add_feature(vwap, group='microstructure')
```

If `round_params['feature_groups']` is present, only transforms whose group is in that string are applied. Ungrouped transforms always run. The value is a pipe-delimited string of active group names, or `'all'` to include every group.

### Conditional inclusion

Use `include_if=` when a transform should only run if a boolean round parameter is true.

```python
.add_feature(vwap, include_if='use_vwap')
```

## Parameter-Controlled Perturbations

The manifest builder now supports several perturbation-style workflows directly in the declarative surface.

### Feature-group selection

Use `group=` on indicators or features, then pass `feature_groups` in `round_params`.

```python
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

```python
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

On a live local prep run in this repo, `feature_drop_count=1` and `feature_drop_seed=42` produced:

```python
round_params['_dropped_features'] == ['vol_5']
```

In artifact-rich runs, that `_dropped_features` payload is stored into `round_data.jsonl`, and [Trainer](Trainer.md) reproduces the same drop set during promotion.

### Scaler choice from parameters

Use `set_scaler_from_params()` when scaler selection itself is part of the search space.

```python
manifest.set_scaler_from_params('scaler_type')

params = {
    'scaler_type': ['logreg', 'robust', 'rank_gauss'],
}
```

On live local manifest-prep runs in this repo, this resolved correctly to:

- `RobustScaler` when `scaler_type='robust'`
- `RankGaussScaler` when `scaler_type='rank_gauss'`

### Manifest-level overrides

Use `with_params_override(...)` when you want a manifest clone with structural changes that should not come from the round search itself.

Typical examples:

- `split_config=(1, 0, 0)` for full-data retraining in [Trainer](Trainer.md)
- `start_date_limit=...` for a controlled data-window variant
- `n_rows=...` for test or smoke paths

Use round parameters for search-time variation and `with_params_override(...)` for external structural control.

Then in `params()`:

```python
{
    'use_vwap': [True, False],
}
```

## Target Configuration

Target construction is handled through `with_target_label(...)`.

```python
.with_target_label('quantile_flag')
    ...
    .done()
```

The target column is placed last before Limen finalizes the `data_dict`.

### `add_fitted_transform(func)`

Use fitted transforms when a target step needs a value computed on the training split first, then reused on validation and test.

```python
.with_target_label('quantile_flag')
    .add_fitted_transform(quantile_flag)
        .fit_param('_quantile_cutoff', compute_quantile_cutoff, col='roc_{roc_period}', q='q')
        .with_params(col='roc_{roc_period}', cutoff='_quantile_cutoff')
    .done()
```

In that example:

1. `compute_quantile_cutoff(train_data, col='roc_12', q=0.35)` runs on train only
2. the result is stored as `_quantile_cutoff`
3. `quantile_flag(..., cutoff=_quantile_cutoff)` is applied to train, validation, and test

This is the manifest-safe way to compute train-only thresholds.

### `add_transform(func, **params)`

Use plain transforms when the step does not need fitted state.

```python
.with_target_label('quantile_flag')
    .add_transform(shift_column_transform, shift='shift', column='target_column')
    .done()
```

`target_column` is injected automatically so you can reference the current target name inside the transform block.

### `done()`

Ends the target builder and returns the manifest for further chaining.

## Scaling

### `set_scaler(transform_class, param_name='_scaler')`

Configure a fitted scaler that is instantiated on train and then applied across the splits.

```python
from limen.scalers import LogRegScaler

.set_scaler(LogRegScaler)
```

The fitted scaler is stored in the resulting `data_dict` under `_scaler`.

### `set_scaler_from_params(param_name='scaler_type')`

Select a scaler dynamically from `SCALER_REGISTRY` using a round parameter.

```python
.set_scaler_from_params('scaler_type')
```

Then in `params()`:

```python
{
    'scaler_type': ['logreg', 'linear', 'robust'],
}
```

Use this when scaler choice is itself part of the search space.

## Rule-Based Strategy

Use `with_strategy()` when the strategy is expressed as boolean predicate logic over indicator columns — no Python model code required. This path is for rule-based SFDs only and produces a different `data_dict` shape than the ML path.

### `with_strategy(conditions, entry)`

Configure a rule-based strategy from a list of condition config dicts and an entry signal id.

```python
from limen.sfd.reference_architecture.rule_based import rule_based

conditions = [
    {'id': 'rsi_oversold', 'type': 'threshold', 'column': 'wilder_rsi_{rsi_period}', 'operator': '<', 'value': '{rsi_threshold}'},
    {'id': 'above_ema',    'type': 'relative',   'column': 'close', 'operator': '>', 'other_column': 'ema_{ema_period}'},
    {'id': 'entry',        'operator': 'and',     'operands': ['rsi_oversold', 'above_ema']},
]

manifest = (
    Manifest()
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
        'conditions': [...],  # condition dicts as configured
        'entry': 'entry',
    },
    '_alignment': {...},
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

```python
from limen.sfd.reference_architecture import logreg_binary

.with_reference_architecture(logreg_binary)
```

The model function must accept the prepared `data_dict` as its first argument.

All remaining parameters are auto-mapped from `round_params` by signature inspection. For example, if your model function has:

```python
def logreg_binary(data, C=1.0, class_weight=None, max_iter=100, solver='lbfgs'):
    ...
```

then manifest execution will automatically pull `C`, `class_weight`, `max_iter`, and `solver` from the current round when those keys exist in `round_params`.

Required model parameters with no defaults must be present in the round params or Limen will raise.

## Feature Ablation

### `set_feature_ablation(drop_count_key='feature_drop_count', seed_key='feature_drop_seed')`

Configure deterministic drop-N feature ablation after target construction.

```python
.set_feature_ablation()
```

Then in `params()`:

```python
{
    'feature_drop_count': [0, 1, 2],
    'feature_drop_seed': [42],
}
```

This is useful when you want feature robustness to be part of the search itself.

## Data Dict Extension

### `add_to_data_dict(func)`

Append custom entries to the finalized `data_dict`.

```python
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

This is especially useful in retraining workflows where you want to promote a winning round from train/val/test mode into all-data training.

Supported override behavior today:

- `split_config=(...)` overrides the split ratios
- other keys are interpreted as data-source parameter overrides and are validated against the configured data-source method signature

Example:

```python
manifest_small = manifest.with_params_override(n_rows=1000)
manifest_full = manifest.with_params_override(split_config=(1, 0, 0))
```

The original manifest remains unchanged.

## What `prepare_data()` Produces

The manifest ultimately builds Limen's standard `data_dict`.

From a live local preparation pass on the foundational logistic-regression manifest, the resulting keys were:

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

- input: frame with the columns your transform expects
- output: frame with the new columns added
- behavior: deterministic with respect to the resolved parameters

If you are adding custom manifest transforms, write them in the same style as Limen's built-ins: accept a frame plus keyword arguments, and return the transformed frame.

### Fitted parameter computation functions

A fitted-parameter compute function is called only on the training split.

The contract is:

- input: eager `pl.DataFrame` plus resolved kwargs
- output: one fitted value

That fitted value is then stored under the name you gave in `fit_param(...)` and can be referenced later in the target or scaler step.

### Target transform functions

Target transforms operate on eager `pl.DataFrame` objects after the feature stage.

The contract is:

- input: eager `pl.DataFrame` plus resolved kwargs
- output: transformed `pl.DataFrame`

Use fitted transforms when the function depends on train-only fitted state. Use plain transforms otherwise.

### Model functions

The architecture function configured in `with_reference_architecture(...)` must:

- accept the finalized `data_dict` as its first argument
- accept any searched model parameters as named kwargs
- return a results dictionary compatible with Limen metrics and logging

If the model returns `_preds`, UEL stores them in `uel.preds` and the `Log` layer can use them for post-run analysis.

## Common Recipes

### Binary target from a fitted quantile cutoff

This is the most important target recipe in the current Limen style:

```python
.with_target_label('quantile_flag')
    .add_fitted_transform(quantile_flag)
        .fit_param('_cutoff', compute_quantile_cutoff, col='roc_{roc_period}', q='q')
        .with_params(col='roc_{roc_period}', cutoff='_cutoff')
    .add_transform(shift_column_transform, shift='shift', column='target_column')
    .done()
```

### Search over feature families

```python
def params():
    return {
        'feature_groups': ['momentum', 'momentum|microstructure'],
        'roc_period': [4, 8, 12],
    }

manifest = (
    Manifest()
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

manifest = Manifest().set_scaler_from_params('scaler_type')
```

### Retrain on all data

```python
manifest_full = manifest.with_params_override(split_config=(1, 0, 0))
```

That pattern is central to [Trainer](Trainer.md).

## Read Next

- Continue to [Universal Experiment Loop](Universal-Experiment-Loop.md) to run a manifest-driven SFD.
- Continue to [Historical Data](Historical-Data.md) if you need the data surfaces a manifest can point at.
- Continue to [Data Bars](Data-Bars.md) if bar formation is part of the experiment design.
- Use [Indicators](Indicators.md), [Features](Features.md), [Transforms](Transforms.md), and [Scalers](Scalers.md) as the reference layer while authoring manifests.
