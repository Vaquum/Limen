# Targets

Target classes in Limen define how the target column is built during the manifest preparation pipeline. A target class is fitted once on the training split, then applied to validation and test without refitting.

Use this page to choose a target class, understand the convention each class must follow, or see what columns each class adds.

## How Targets Fit In The Pipeline

The manifest pipeline is split-first:

1. fetch and prepare raw data
2. split into train, validation, and test
3. apply indicators and features
4. fit the target class on the training split
5. apply the fitted instance to all splits using the stored state

This ensures that any statistics derived from the training data (such as quantile cutoffs) do not leak into validation or test.

## Using A Target Class

```python
from limen.targets import QuantileBinaryTarget

manifest.with_target_label(
    'quantile_flag',
    QuantileBinaryTarget,
    fit_params={'source_column': 'roc_{roc_period}', 'quantile': 'q'},
    transform_params={'shift': 'shift'},
)
```

`fit_params` are forwarded to `__init__` on the training split. `transform_params` are forwarded to `transform()` on every split. Both support round-param references such that `'q'` resolves to the current round's `q` value.

## Choosing A Target Class

| Class | Output type | Fitted on train | Shift default | Notes |
|---|---|---|---|---|
| `QuantileBinaryTarget` | binary `UInt8` | yes — quantile cutoff | `shift=0` | Positive label above the top-N quantile of the source column. |
| `ThresholdBinaryTarget` | binary `UInt8` | no | `shift=-1` | Positive label above a fixed numeric threshold. |
| `TradelineLongBinaryTarget` | binary `UInt8` | yes — line-height percentile threshold | n/a | Confirmed long breakout: forward max over `lookahead_hours` reaches the train-fitted threshold and a point return confirms it. |
| `ForwardBreakoutTarget` | binary `UInt8` | no | `shift=-1` | Positive label if price rises at least `threshold` over the next `forward_periods` bars. |
| `EmaBreakoutTarget` | binary `UInt8` | no | n/a | Positive label if price `breakout_horizon` bars ahead exceeds EMA by `breakout_delta`. |
| `NextBarUpTarget` | binary `UInt8` | no | none | Canonical up decoder outcome: positive label if the next close is higher. |
| `NextBarDownTarget` | binary `UInt8` | no | none | Canonical down decoder outcome: positive label if the next close is lower. |
| `NextReturnTarget` | continuous `Float64` | no | n/a | Percentage return over the next N bars. Use with regression architectures. |
| `VolNormalizedReturnTarget` | continuous `Float64` | yes — train sanity gate | none | Canonical return decoder outcome: log return divided by prior Parkinson volatility. |
| `ForwardVolNormalizedReturnTarget` | continuous `Float64` | yes — train sanity gate | none | Predictive label: forward log return divided by current Parkinson volatility. |
| `TripleBarrierTarget` | ternary `Int8` | no | n/a | First barrier touched over the next `max_horizon` bars: `1` upper, `-1` lower, `0` vertical. Horizontal barriers are volatility multiples. |
| `RiskRewardRatioTarget` | continuous `Float64` | no | n/a | Ratio of `capturable_breakout` to absolute `max_drawdown` per row. |
| `ExitQualityTarget` | continuous `Float64` | no | n/a | Categorical score for closed trades based on `exit_reason` and `exit_net_return`. |
| `RandomBinaryTarget` | binary `UInt8` | no | none | Uniformly random labels. Use as a noise benchmark. |
| `IdentityTarget` | existing column | no | none | Target column already present in the data. Validates the column exists on every split. |

## Reference

### `QuantileBinaryTarget`

```python
QuantileBinaryTarget(train_data, target_name, source_column, quantile)
```

Fits a quantile cutoff on the training split. Labels a bar as positive if the source column exceeds the `(1 - quantile)` quantile of the training distribution.

```python
.with_target_label(
    'quantile_flag',
    QuantileBinaryTarget,
    fit_params={'source_column': 'roc_4', 'quantile': 0.35},
    transform_params={'shift': -1},
)
```

| Parameter | Type | Description |
|---|---|---|
| `source_column` | `str` | Column used to compute the quantile threshold |
| `quantile` | `float` | Top-N fraction for positive label (`0.3` = top 30%) |
| `shift` | `int` | Periods to shift the label; `shift=0` means no shift |

### `TradelineLongBinaryTarget`

```python
TradelineLongBinaryTarget(train_data, target_name, max_duration_hours, min_height_pct, long_threshold_percentile)
```

Fits the breakout threshold on the training split: detects price lines on `close` (`limen.utils.find_price_lines`) and stores the `long_threshold_percentile` percentile of long-line heights. Raises when the training split yields no long lines. The transform labels a bar `1` when the maximum close over `[t, t + lookahead_hours]` reaches the threshold AND the point return at `+confirmation_hours` or at `+lookahead_hours` also exceeds it — the move must stick, not just spike. Rows without a full forward window become null and fall to the pipeline's `drop_nulls`.

```python
.with_target_label(
    'tradeline_long',
    TradelineLongBinaryTarget,
    fit_params={'max_duration_hours': 48, 'min_height_pct': 0.003, 'long_threshold_percentile': 75},
    transform_params={'lookahead_hours': 48, 'confirmation_hours': 24},
)
```

| Parameter | Type | Description |
|---|---|---|
| `max_duration_hours` | `int` | Exclusive upper bound on line duration in bars (fit) |
| `min_height_pct` | `float` | Minimum absolute line height as a fraction of start price (fit) |
| `long_threshold_percentile` | `float` | Percentile of long-line heights used as the threshold, in `[0, 100]` (fit) |
| `lookahead_hours` | `int` | Forward window for the breakout check; default `48` |
| `confirmation_hours` | `int` | Forward offset for the point-confirmation check; default `24` |

### `ThresholdBinaryTarget`

```python
ThresholdBinaryTarget(train_data, target_name, source_column, threshold)
```

Labels a bar as positive if the source column exceeds a fixed threshold. No fitting step — the threshold is specified directly.

```python
.with_target_label(
    'above_zero',
    ThresholdBinaryTarget,
    fit_params={'source_column': 'roc_4', 'threshold': 0.0},
    transform_params={'shift': -1},
)
```

| Parameter | Type | Description |
|---|---|---|
| `source_column` | `str` | Column to compare against the threshold |
| `threshold` | `float` | Fixed value; bars above this are labeled 1 |
| `shift` | `int` | Periods to shift the label; default `-1` |

### `ForwardBreakoutTarget`

```python
ForwardBreakoutTarget(train_data, target_name)
```

Labels a bar as positive if the close price rises by at least `threshold` over the next `forward_periods` bars. No fitting step.

```python
.with_target_label(
    'breakout',
    ForwardBreakoutTarget,
    transform_params={'forward_periods': 24, 'threshold': 0.02, 'shift': -1},
)
```

| Parameter | Type | Description |
|---|---|---|
| `forward_periods` | `int` | Look-ahead window in bars; default `24` |
| `threshold` | `float` | Minimum return for positive label; default `0.02` (2%) |
| `shift` | `int` | Additional shift applied after labeling; default `-1` |

### `EmaBreakoutTarget`

```python
EmaBreakoutTarget(train_data, target_name)
```

Labels a bar as positive if the price `breakout_horizon` bars ahead exceeds the EMA of `target_col` by at least `breakout_delta`. No fitting step.

```python
.with_target_label(
    'breakout_ema',
    EmaBreakoutTarget,
    transform_params={'target_col': 'close', 'ema_span': 30, 'breakout_delta': 0.2, 'breakout_horizon': 3},
)
```

| Parameter | Type | Description |
|---|---|---|
| `target_col` | `str` | Column name to analyze for breakouts; default `'close'` |
| `ema_span` | `int` | Period for EMA calculation; default `30` |
| `breakout_delta` | `float` | Minimum EMA-relative displacement for positive label; default `0.2` |
| `breakout_horizon` | `int` | Forward window in bars; default `3` |

### `NextReturnTarget`

```python
NextReturnTarget(train_data, target_name)
```

Produces a continuous target as the percentage return over the next N bars. Use with regression architectures such as `xgboost_regressor`. No fitting step.

```python
.with_target_label(
    'next_return',
    NextReturnTarget,
    transform_params={'periods': 1, 'scale': 100.0},
)
```

| Parameter | Type | Description |
|---|---|---|
| `periods` | `int` | Look-ahead window in bars; default `1` |
| `scale` | `float` | Multiplier applied to the raw return; default `100.0` |

### `NextBarUpTarget`

```python
NextBarUpTarget(train_data, target_name)
```

Produces a binary target named `next_bar_up`: `1` when the next close is higher than the current close, `0` otherwise. The final row in each split has no next bar, so it is null and is dropped by manifest preparation.

```python
.with_target_label('next_bar_up', NextBarUpTarget)
```

No parameters.

### `NextBarDownTarget`

```python
NextBarDownTarget(train_data, target_name)
```

Produces a binary target named `next_bar_down`: `1` when the next close is lower than the current close, `0` otherwise. The final row in each split has no next bar, so it is null and is dropped by manifest preparation.

```python
.with_target_label('next_bar_down', NextBarDownTarget)
```

No parameters.

### `VolNormalizedReturnTarget`

```python
VolNormalizedReturnTarget(train_data, target_name, high_col='high', low_col='low', open_col='open', close_col='close', halflife=50, min_periods=150)
```

Produces a continuous target named `vol_normalized_return`. The target is `log(close / close.shift(1)) / sigma.shift(1)`, where `sigma` is the EWMA Parkinson volatility from `(log(high / low) ** 2) / (4 * ln(2))`. The one-bar sigma shift prevents the current bar range from entering the current return label.

Dirty OHLC rows are dropped before target construction when `high < max(open, close)` or `low > min(open, close)`. Zero volatility is converted to null, not epsilon.

```python
.with_target_label(
    'vol_normalized_return',
    VolNormalizedReturnTarget,
    fit_params={'halflife': 50, 'min_periods': 150},
)
```

| Parameter | Type | Description |
|---|---|---|
| `high_col` | `str` | High price column; default `'high'` |
| `low_col` | `str` | Low price column; default `'low'` |
| `open_col` | `str` | Open price column used for dirty-bar filtering; default `'open'` |
| `close_col` | `str` | Close price column; default `'close'` |
| `halflife` | `int` | EWMA half-life in bars; default `50` |
| `min_periods` | `int` | Warmup before volatility is emitted; default `150` |

The training split must pass the Parkinson-to-close volatility sanity gate: median `sigma_P / sigma_C2C` must be within `[0.9, 1.1]`. Bucketing is deliberately downstream: fit `q33`/`q66` of `abs(vol_normalized_return)` on train, persist the boundaries, and load them at inference instead of refitting online.

### `ForwardVolNormalizedReturnTarget`

```python
ForwardVolNormalizedReturnTarget(train_data, target_name, periods=1, absolute=False, halflife=50, min_periods=150, high_col='high', low_col='low', open_col='open', close_col='close')
```

Produces a continuous predictive label as `log(close.shift(-periods) / close) / sigma`, where `sigma` is the current EWMA Parkinson volatility. Set `absolute=True` to normalize absolute forward movement instead of signed direction.

```python
.with_target_label(
    'forward_vol_normalized_return',
    ForwardVolNormalizedReturnTarget,
    fit_params={'periods': 1, 'halflife': 50, 'min_periods': 150},
)
```

| Parameter | Type | Description |
|---|---|---|
| `periods` | `int` | Forward return horizon in bars; default `1` |
| `absolute` | `bool` | Whether to use absolute forward log return; default `False` |
| `halflife` | `int` | EWMA half-life in bars; default `50` |
| `min_periods` | `int` | Warmup before volatility is emitted; default `150` |
| `high_col` | `str` | High price column; default `'high'` |
| `low_col` | `str` | Low price column; default `'low'` |
| `open_col` | `str` | Open price column used for dirty-bar filtering; default `'open'` |
| `close_col` | `str` | Close price column; default `'close'` |

The class reuses the same dirty-bar filter and Parkinson-to-close volatility sanity gate as `VolNormalizedReturnTarget`.

### `TripleBarrierTarget`

```python
TripleBarrierTarget(train_data, target_name, upper_multiple=1.0, lower_multiple=1.0, max_horizon=10, span=100, min_periods=100, close_col='close')
```

Produces a ternary label from Lopez de Prado's triple-barrier method. For each bar, the forward close path is followed for up to `max_horizon` bars and labelled by the first barrier it touches: `1` if the upper (profit-taking) barrier is touched first, `-1` if the lower (stop-loss) barrier is touched first, and `0` if neither is touched within `max_horizon` bars (the vertical barrier). No fitting step.

The horizontal barriers are volatility multiples — `+upper_multiple * sigma` and `-lower_multiple * sigma`, where `sigma` is the EWMA standard deviation of close-to-close returns with the given `span`. The barrier width is taken from the entry-bar volatility, so it adapts to the local regime instead of using a fixed percentage.

A bar is null during the volatility warmup (`min_periods`), when its volatility is zero, and when the vertical barrier would extend past the available data and no barrier is touched within the truncated window. Manifest preparation drops null target rows. Size `min_periods` (and `span`) against the smallest split: a split shorter than the volatility warmup is entirely null and is dropped, leaving no labelled rows for that split.

```python
.with_target_label(
    'triple_barrier',
    TripleBarrierTarget,
    fit_params={'upper_multiple': 2.0, 'lower_multiple': 2.0, 'max_horizon': 24, 'span': 100, 'min_periods': 100},
)
```

| Parameter | Type | Description |
|---|---|---|
| `upper_multiple` | `float` | Profit-taking barrier width in volatility units; default `1.0` |
| `lower_multiple` | `float` | Stop-loss barrier width in volatility units; default `1.0` |
| `max_horizon` | `int` | Vertical barrier as a forward window in bars; default `10` |
| `span` | `int` | EWMA span for the close-to-close return volatility; default `100` |
| `min_periods` | `int` | Warmup before volatility is emitted; default `100` |
| `close_col` | `str` | Close price column used for the barrier path; default `'close'` |

### `RiskRewardRatioTarget`

```python
RiskRewardRatioTarget(train_data, target_name)
```

Computes `capturable_breakout / (|max_drawdown| + 0.001)` per row. Expects both columns to be available on every split. No fitting step.

```python
.with_target_label('rr_ratio', RiskRewardRatioTarget)
```

No parameters.

### `ExitQualityTarget`

```python
ExitQualityTarget(train_data, target_name)
```

Scores closed trades as high, low, or medium based on `exit_reason` and `exit_net_return`. Expects both columns to be available on every split. No fitting step.

```python
.with_target_label(
    'exit_quality',
    ExitQualityTarget,
    transform_params={'exit_quality_high': 1.0, 'exit_quality_low': 0.2, 'exit_quality_medium': 0.5},
)
```

| Parameter | Type | Description |
|---|---|---|
| `exit_quality_high` | `float` | Score for profitable `target_hit` or `trailing_stop` exits; default `1.0` |
| `exit_quality_low` | `float` | Score for `stop_loss` or unprofitable `timeout` exits; default `0.2` |
| `exit_quality_medium` | `float` | Score for neutral exits; default `0.5` |

### `RandomBinaryTarget`

```python
RandomBinaryTarget(train_data, target_name)
```

Produces uniformly random binary labels. No fitting step. Use as a noise benchmark to verify that a trained model outperforms chance.

```python
.with_target_label('outcome', RandomBinaryTarget)
```

No parameters.

### `IdentityTarget`

```python
IdentityTarget(train_data, target_name)
```

For when the target column is already present in the data. Validates that the column exists on the training split and on every subsequent split. The data is returned unchanged; no new column is added.

```python
.with_target_label('label', IdentityTarget)
```

No parameters.

## Writing A Custom Target Class

A target class must follow this convention:

```python
class MyTarget:

    def __init__(self, train_data: pl.DataFrame, target_name: str, **fit_params) -> None:
        self.target_name = target_name
        # fit on train_data here if needed

    def transform(self, data: pl.DataFrame, **transform_params) -> pl.DataFrame:
        return data.with_columns(
            # ... compute label ...
            .alias(self.target_name)
        )
```

There is no base class to inherit from — the convention is enforced structurally, the same way scalers work. Any class that accepts `(train_data, target_name, **fit_params)` in `__init__` and exposes `transform(data, **transform_params) -> pl.DataFrame` is a valid target class.

## Read Next

- [Experiment Manifest](Experiment-Manifest.md) for how targets plug into the split-first pipeline
- [Scalers](Scalers.md) for train-fitted preprocessing that runs after target construction
- [Transforms](Transforms.md) for stateless post-model helpers
