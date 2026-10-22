# Targets

A target defines the observation a model learns to predict. Python Research builds labels in caller-owned preparation. Compiled Research fits a target class on training observations and applies its stored state to later splits.

Use this page to choose a target class, understand the convention each class must follow, or see what columns each class adds.

## How Targets Fit In The Pipeline

The manifest pipeline is split-first:

1. fetch and prepare raw data
2. split into train, validation, and test
3. apply indicators and features
4. fit the target class on the training split
5. apply the fitted instance to all splits using the stored state

This ensures that any statistics derived from the training data (such as quantile cutoffs) do not leak into validation or test.

## One outcome, one preparation boundary

The [Python project](examples/python-research/README.md) predicts next-bar open-to-close return. Its preparation function runs separately inside each split. The negative shift creates the future outcome used for fitting and validation; that outcome is excluded from inference features.

<!-- example: targets-python-next-return source=docs/examples/python-research/research.py region=next_return_target scenario=selection interface=python role=executed -->
```python
def prepare_split(bars, round_params):
    return features(bars, round_params["window"]).with_columns(
        (pl.col("close").shift(-1) / pl.col("open").shift(-1) - 1)
        .alias("next_return")
    ).slice(12).drop_nulls(["momentum", "next_return"])
```

The [Compiled project](examples/compiled-research/README.md) predicts whether the next momentum observation exceeds a training-fitted quantile. Its Manifest declares the class and the constructor and transform arguments:

<!-- example: targets-compiled-quantile source=docs/examples/compiled-research/selection.yaml region=fitted_target scenario=selection interface=compiled role=executed -->
```yaml
    target:
      name: quantile_flag
      class: limen.targets.QuantileBinaryTarget
      fit_params:
        source_column: roc_1
        quantile: 0.5
      transform_params:
        shift: -1
```

`fit_params` go to the target constructor on the training split; `transform_params` go to `transform()` on each split. Values can refer to declared round parameters. Compiled Research declares these values in YAML. Python Research can call the same target classes directly, provided its preparation owns the fitting and split boundaries.

The reference below gives current package signatures and the behavior of each alternative. Constructor and transform signatures describe the API; they do not claim that every target participates in the two selection runs. Data-dependent targets still require their stated columns, fitting observations and execution economics.

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
| `TradeOutcomeTarget` | binary `Int8` or net return `Float64` | no | none | Simulates independent long/short trades under the manifest's execution rules; incomplete outcomes stay null. |

## Reference

### `QuantileBinaryTarget`

<!-- reference: target-quantile-binary-target-constructor source=limen/targets/quantile_binary.py region=signature:QuantileBinaryTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str, source_column: str, quantile: float) -> None:
    ...
```

Fits a quantile cutoff on the training split. Labels a bar as positive if the source column exceeds the `(1 - quantile)` quantile of the training distribution.

<!-- reference: target-quantile-binary-target-transform source=limen/targets/quantile_binary.py region=signature:QuantileBinaryTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame, shift: int=0) -> pl.DataFrame:
    ...
```

| Parameter | Type | Description |
|---|---|---|
| `source_column` | `str` | Column used to compute the quantile threshold |
| `quantile` | `float` | Top-N fraction for positive label (`0.3` = top 30%) |
| `shift` | `int` | Periods to shift the label; `shift=0` means no shift |

### `TradelineLongBinaryTarget`

<!-- reference: target-tradeline-long-binary-target-constructor source=limen/targets/tradeline_long_binary.py region=signature:TradelineLongBinaryTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str, max_duration_hours: int, min_height_pct: float, long_threshold_percentile: float) -> None:
    ...
```

Fits the breakout threshold on the training split: detects price lines on `close` (`limen.utils.find_price_lines`) and stores the `long_threshold_percentile` percentile of long-line heights. Raises when the training split yields no long lines. The transform labels a bar `1` when the maximum close over `[t, t + lookahead_hours]` reaches the threshold AND the point return at `+confirmation_hours` or at `+lookahead_hours` also exceeds it — the move must stick, not just spike. Rows without a full forward window become null and fall to the pipeline's `drop_nulls`.

<!-- reference: target-tradeline-long-binary-target-transform source=limen/targets/tradeline_long_binary.py region=signature:TradelineLongBinaryTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame, lookahead_hours: int=DEFAULT_LOOKAHEAD_HOURS, confirmation_hours: int=DEFAULT_CONFIRMATION_HOURS) -> pl.DataFrame:
    ...
```

| Parameter | Type | Description |
|---|---|---|
| `max_duration_hours` | `int` | Exclusive upper bound on line duration in bars (fit) |
| `min_height_pct` | `float` | Minimum absolute line height as a fraction of start price (fit) |
| `long_threshold_percentile` | `float` | Percentile of long-line heights used as the threshold, in `[0, 100]` (fit) |
| `lookahead_hours` | `int` | Forward window for the breakout check; default `48` |
| `confirmation_hours` | `int` | Forward offset for the point-confirmation check; default `24` |

### `ThresholdBinaryTarget`

<!-- reference: target-threshold-binary-target-constructor source=limen/targets/threshold_binary.py region=signature:ThresholdBinaryTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str, source_column: str, threshold: float) -> None:
    ...
```

Labels a bar as positive if the source column exceeds a fixed threshold. No fitting step — the threshold is specified directly.

<!-- reference: target-threshold-binary-target-transform source=limen/targets/threshold_binary.py region=signature:ThresholdBinaryTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame, shift: int=-1) -> pl.DataFrame:
    ...
```

| Parameter | Type | Description |
|---|---|---|
| `source_column` | `str` | Column to compare against the threshold |
| `threshold` | `float` | Fixed value; bars above this are labeled 1 |
| `shift` | `int` | Periods to shift the label; default `-1` |

### `ForwardBreakoutTarget`

<!-- reference: target-forward-breakout-target-constructor source=limen/targets/forward_breakout.py region=signature:ForwardBreakoutTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:
    ...
```

Labels a bar as positive if the close price rises by at least `threshold` over the next `forward_periods` bars. No fitting step.

<!-- reference: target-forward-breakout-target-transform source=limen/targets/forward_breakout.py region=signature:ForwardBreakoutTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame, forward_periods: int=24, threshold: float=0.02, shift: int=-1) -> pl.DataFrame:
    ...
```

| Parameter | Type | Description |
|---|---|---|
| `forward_periods` | `int` | Look-ahead window in bars; default `24` |
| `threshold` | `float` | Minimum return for positive label; default `0.02` (2%) |
| `shift` | `int` | Additional shift applied after labeling; default `-1` |

### `EmaBreakoutTarget`

<!-- reference: target-ema-breakout-target-constructor source=limen/targets/ema_breakout.py region=signature:EmaBreakoutTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:
    ...
```

Labels a bar as positive if the price `breakout_horizon` bars ahead exceeds the EMA of `target_col` by at least `breakout_delta`. No fitting step.

<!-- reference: target-ema-breakout-target-transform source=limen/targets/ema_breakout.py region=signature:EmaBreakoutTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame, target_col: str='close', ema_span: int=30, breakout_delta: float=0.2, breakout_horizon: int=3) -> pl.DataFrame:
    ...
```

| Parameter | Type | Description |
|---|---|---|
| `target_col` | `str` | Column name to analyze for breakouts; default `'close'` |
| `ema_span` | `int` | Period for EMA calculation; default `30` |
| `breakout_delta` | `float` | Minimum EMA-relative displacement for positive label; default `0.2` |
| `breakout_horizon` | `int` | Forward window in bars; default `3` |

### `NextReturnTarget`

<!-- reference: target-next-return-target-constructor source=limen/targets/next_return.py region=signature:NextReturnTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:
    ...
```

Produces a continuous target as the percentage return over the next N bars. Use with regression architectures such as `xgboost_regressor`. No fitting step.

<!-- reference: target-next-return-target-transform source=limen/targets/next_return.py region=signature:NextReturnTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame, periods: int=1, scale: float=100.0) -> pl.DataFrame:
    ...
```

| Parameter | Type | Description |
|---|---|---|
| `periods` | `int` | Look-ahead window in bars; default `1` |
| `scale` | `float` | Multiplier applied to the raw return; default `100.0` |

### `NextBarUpTarget`

<!-- reference: target-next-bar-up-target-constructor source=limen/targets/next_bar_up.py region=signature:NextBarUpTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:
    ...
```

Produces a binary target named `next_bar_up`: `1` when the next close is higher than the current close, `0` otherwise. The final row in each split has no next bar, so it is null and is dropped by manifest preparation.

<!-- reference: target-next-bar-up-target-transform source=limen/targets/next_bar_up.py region=signature:NextBarUpTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame) -> pl.DataFrame:
    ...
```

No parameters.

### `NextBarDownTarget`

<!-- reference: target-next-bar-down-target-constructor source=limen/targets/next_bar_down.py region=signature:NextBarDownTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:
    ...
```

Produces a binary target named `next_bar_down`: `1` when the next close is lower than the current close, `0` otherwise. The final row in each split has no next bar, so it is null and is dropped by manifest preparation.

<!-- reference: target-next-bar-down-target-transform source=limen/targets/next_bar_down.py region=signature:NextBarDownTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame) -> pl.DataFrame:
    ...
```

No parameters.

### `VolNormalizedReturnTarget`

<!-- reference: target-vol-normalized-return-target-constructor source=limen/targets/vol_normalized_return.py region=signature:VolNormalizedReturnTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str, high_col: str='high', low_col: str='low', open_col: str='open', close_col: str='close', halflife: int=50, min_periods: int=150) -> None:
    ...
```

Produces a continuous target named `vol_normalized_return`. The target is `log(close / close.shift(1)) / sigma.shift(1)`, where `sigma` is the EWMA Parkinson volatility from `(log(high / low) ** 2) / (4 * ln(2))`. The one-bar sigma shift prevents the current bar range from entering the current return label.

Dirty OHLC rows are dropped before target construction when `high < max(open, close)` or `low > min(open, close)`. Zero volatility is converted to null, not epsilon.

<!-- reference: target-vol-normalized-return-target-transform source=limen/targets/vol_normalized_return.py region=signature:VolNormalizedReturnTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame) -> pl.DataFrame:
    ...
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

<!-- reference: target-forward-vol-normalized-return-target-constructor source=limen/targets/forward_vol_normalized_return.py region=signature:ForwardVolNormalizedReturnTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str, periods: int=1, absolute: bool=False, halflife: int=50, min_periods: int=150, high_col: str='high', low_col: str='low', open_col: str='open', close_col: str='close') -> None:
    ...
```

Produces a continuous predictive label as `log(close.shift(-periods) / close) / sigma`, where `sigma` is the current EWMA Parkinson volatility. Set `absolute=True` to normalize absolute forward movement instead of signed direction.

<!-- reference: target-forward-vol-normalized-return-target-transform source=limen/targets/forward_vol_normalized_return.py region=signature:ForwardVolNormalizedReturnTarget.transform role=signature -->
```python
@override
def transform(self, data: pl.DataFrame) -> pl.DataFrame:
    ...
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

<!-- reference: target-triple-barrier-target-constructor source=limen/targets/triple_barrier.py region=signature:TripleBarrierTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str, upper_multiple: float=1.0, lower_multiple: float=1.0, max_horizon: int=10, span: int=100, min_periods: int=100, close_col: str='close') -> None:
    ...
```

Produces a ternary label from Lopez de Prado's triple-barrier method. For each bar, the forward close path is followed for up to `max_horizon` bars and labelled by the first barrier it touches: `1` if the upper (profit-taking) barrier is touched first, `-1` if the lower (stop-loss) barrier is touched first, and `0` if neither is touched within `max_horizon` bars (the vertical barrier). No fitting step.

The horizontal barriers are volatility multiples — `+upper_multiple * sigma` and `-lower_multiple * sigma`, where `sigma` is the EWMA standard deviation of close-to-close returns with the given `span`. The barrier width is taken from the entry-bar volatility, so it adapts to the local regime instead of using a fixed percentage.

A bar is null during the volatility warmup (`min_periods`), when its volatility is zero, and when the vertical barrier would extend past the available data and no barrier is touched within the truncated window. Manifest preparation drops null target rows. Size `min_periods` (and `span`) against the smallest split: a split shorter than the volatility warmup is entirely null and is dropped, leaving no labelled rows for that split.

<!-- reference: target-triple-barrier-target-transform source=limen/targets/triple_barrier.py region=signature:TripleBarrierTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame) -> pl.DataFrame:
    ...
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

<!-- reference: target-risk-reward-ratio-target-constructor source=limen/targets/risk_reward_ratio.py region=signature:RiskRewardRatioTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:
    ...
```

Computes `capturable_breakout / (|max_drawdown| + 0.001)` per row. Expects both columns to be available on every split. No fitting step. No Limen indicator or feature produces `capturable_breakout` or `max_drawdown` — they must be user-supplied in the input data, and `transform` raises `ValueError` when they are absent.

<!-- reference: target-risk-reward-ratio-target-transform source=limen/targets/risk_reward_ratio.py region=signature:RiskRewardRatioTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame) -> pl.DataFrame:
    ...
```

No parameters.

### `ExitQualityTarget`

<!-- reference: target-exit-quality-target-constructor source=limen/targets/exit_quality.py region=signature:ExitQualityTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:
    ...
```

Scores closed trades as high, low, or medium based on `exit_reason` and `exit_net_return`. Expects both columns to be available on every split. No fitting step. No Limen indicator or feature produces `exit_reason` or `exit_net_return` — they must be user-supplied in the input data, and `transform` raises `ValueError` when they are absent.

<!-- reference: target-exit-quality-target-transform source=limen/targets/exit_quality.py region=signature:ExitQualityTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame, exit_quality_high: float=1.0, exit_quality_low: float=0.2, exit_quality_medium: float=0.5) -> pl.DataFrame:
    ...
```

| Parameter | Type | Description |
|---|---|---|
| `exit_quality_high` | `float` | Score for profitable `target_hit` or `trailing_stop` exits; default `1.0` |
| `exit_quality_low` | `float` | Score for `stop_loss` or unprofitable `timeout` exits; default `0.2` |
| `exit_quality_medium` | `float` | Score for neutral exits; default `0.5` |

### `RandomBinaryTarget`

<!-- reference: target-random-binary-target-constructor source=limen/targets/random_binary.py region=signature:RandomBinaryTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str, random_state: int | None=None) -> None:
    ...
```

Produces uniformly random binary labels. No fitting step. Use as a noise benchmark to verify that a trained model outperforms chance.

<!-- reference: target-random-binary-target-transform source=limen/targets/random_binary.py region=signature:RandomBinaryTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame) -> pl.DataFrame:
    ...
```

`random_state` is optional. Pass an integer for reproducible labels across equal split sizes and call order; `None` uses an unseeded generator.

### `IdentityTarget`

<!-- reference: target-identity-target-constructor source=limen/targets/identity.py region=signature:IdentityTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:
    ...
```

For when the target column is already present in the data. Validates that the column exists on the training split and on every subsequent split. The data is returned unchanged; no new column is added.

<!-- reference: target-identity-target-transform source=limen/targets/identity.py region=signature:IdentityTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame) -> pl.DataFrame:
    ...
```

No parameters.

## Simulated Trade Outcomes

`TradeOutcomeTarget` requires configured execution economics and a linear perpetual product supporting both sides. The manifest injects `TradeTargetContext` into construction and every transform; callers do not build it during normal manifest use.

<!-- reference: target-trade-outcome-constructor source=limen/targets/trade_outcome.py region=signature:TradeOutcomeTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str, *, trade_context: TradeTargetContext, side: Literal['long', 'short']='long', output: Literal['binary', 'net_return']='binary', candidate_exposure: float=1.0) -> None:
    ...
```

<!-- reference: target-trade-outcome-transform source=limen/targets/trade_outcome.py region=signature:TradeOutcomeTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame, *, trade_context: TradeTargetContext, **transform_params: object) -> pl.DataFrame:
    ...
```

Each feature-valid bar starts independent long and short candidates with fixed candidate exposure. Entry, TP/SL, elapsed exits, costs, rounding and funding use the [shared execution contract](Backtest.md#signed-exposure-elapsed-exits-and-funding). Returns equal completed net PnL divided by pre-entry equity; binary output is strictly positive return. Zero is unprofitable. At least one exit rule is required.

Only observations within the true split can complete a candidate. A completed late trade remains eligible even after an earlier unfinished candidate. An unfinished outcome is null, with a private availability mask and completion timestamp. Those masks select fitting and metric rows; they never remove otherwise valid inference rows. Warm-up rows outside a split receive no outcome.

The long/short returns, masks and exit times live in `_trade_labels: OutcomeLabels`, bound to the execution contracts. They never enter feature selection, scaling, PCA or Sensor inputs. The public target column contains only the requested side/output. Architectures consuming these targets must explicitly support private availability masks; [direction/sizing](Reference-Architecture.md#direction-and-sizing) provides that integration.

Unknown target parameters, mismatched contracts and missing economics fail before fitting.

## Writing A Custom Target Class

A custom target accepts training data and the target column name at construction, then exposes `transform(data, **transform_params)`. Fit any learned state in the constructor and reuse it in `transform`. `NextBarUpTarget` gives the minimal concrete package interface:

<!-- reference: target-custom-constructor source=limen/targets/next_bar_up.py region=signature:NextBarUpTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:
    ...
```

<!-- reference: target-custom-transform source=limen/targets/next_bar_up.py region=signature:NextBarUpTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame) -> pl.DataFrame:
    ...
```

There is no base class to inherit from — the convention is enforced structurally, the same way scalers work. Any class that accepts `(train_data, target_name, **fit_params)` in `__init__` and exposes `transform(data, **transform_params) -> pl.DataFrame` is a valid target class.

`source_columns` is optional on the target class or fitted instance. Declare a
list of column names consumed to build labels that must not become features.
Manifest preparation removes any declared columns still present after
`transform()`, before scaling and feature extraction, on all three splits and
fitted-target replay. ML and native rule Sensor preparation apply declared
exclusions from fitted target metadata before deciding feature availability,
without recomputing labels or fitting the target. ML scaling and PCA then retain
their existing order. Missing or empty declarations exclude no live columns;
undeclared drops or mutations inside `transform()` are not replayed at inference.
Rule Sensor preparation requires the fitted target instance and raises if it is missing.

A declaration must be a list of strings and must not include `target_name`.
Invalid declarations raise `TypeError`; excluding the target raises `ValueError`.
Other columns, values and retained row order remain unchanged. The declaration
excludes only named inputs; it does not automatically detect look-ahead features.

`ExitQualityTarget` declares `exit_reason` and `exit_net_return`;
`RiskRewardRatioTarget` declares `capturable_breakout` and `max_drawdown`.
Their directly called `transform()` methods retain their existing column removal.
Custom targets may keep their own removal during research or rely on manifest
enforcement; declare those source columns to exclude them during live preparation.
Columns already removed by a research target are accepted.

## Read Next

- [Experiment Manifest](Experiment-Manifest.md) for how targets plug into the split-first pipeline
- [Scalers](Scalers.md) for train-fitted preprocessing that runs after target construction
- [Transforms](Transforms.md) for stateless post-model helpers

Meta allocation retains the primary target. Forward QuantileBinaryTarget fitting is rebuilt inside each chronological prefix; prediction-block membership is feature-valid and includes the tail without completed forward targets. Allocator supervision is separate: signed net PnL divided by the first actual entry fill notional, available only after real closure inside the opening training block. Open, unfilled and crossing opportunities remain unavailable; they are not zero-return training examples. See [Backtest](Backtest.md#learned-allocation-around-a-primary).
