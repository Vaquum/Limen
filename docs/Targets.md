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

## Candidate target perturbations

`ForwardBreakoutTarget.transform` accepts `horizon_jitter`, `threshold_noise_std`, `noise_seed` and `label_smoothing`. `TripleBarrierTarget.transform` accepts `horizon_jitter`, `barrier_noise_std` and `noise_seed`. All default to zero, retaining the original target calculation.

Jitter is drawn once per candidate and reused across its partitions. A SHA-256 seed over the `limen-target-perturbation-v1` namespace, complete permutation ID and explicit `noise_seed` initializes NumPy's generator. The horizon receives a uniform integer offset in the inclusive jitter range; every possible horizon must remain positive. Thresholds/barrier multiples receive independent multiplicative Gaussian draws `base * (1 + Normal(0, std))`. Nonpositive or nonfinite realized thresholds raise errors. The fitted target retains `perturbation_parameters`, including the realized horizon, thresholds and identity.

UEL, parallel workers and Trainer/fold reconstruction supply the logged candidate ID automatically, including runs with context parameters. `permutation_id` and `training` are reserved framework operands in manifest/YAML transforms. Direct target calls must provide an ID when enabling noise; direct manifest preparation derives an identity from its supplied parameters. Declare the noise controls through ordinary target `transform_params` references so their values and seeds remain in the round log. Enabled permutation-seeded noise is refused by `factorize=True`, whose cross-candidate reuse would otherwise change its meaning.

Binary label smoothing applies `(1 - alpha) * y + alpha / 2` only to the training partition, with `0 <= alpha < 1`. Validation and test labels remain hard. The model must accept continuous training targets; native hard-label classifiers reject them through their normal model validation. The recorded proof uses DLinear and checks unchanged evaluation labels, Python/CLI execution, worker parity and Trainer reconstruction.

The recorded proofs are `tests/test_target_perturbations.py`; they also compare noisy targets directly against the original calculation under their realized parameters.

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
def transform(self, data: pl.DataFrame, forward_periods: int=24, threshold: float=0.02, shift: int=-1, *, horizon_jitter: int=0, threshold_noise_std: float=0.0, noise_seed: int=0, label_smoothing: float=0.0, permutation_id: str | None=None, training: bool=False) -> pl.DataFrame:
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
def transform(self, data: pl.DataFrame, *, horizon_jitter: int=0, barrier_noise_std: float=0.0, noise_seed: int=0, permutation_id: str | None=None, training: bool=False) -> pl.DataFrame:
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

## Recorded target variants

The [Python project](examples/python-research/README.md) retains separate train/validation/test label frames in `search/targets/`. Its raw-bar cases also preserve a data-dependent volatility sanity refusal and the required-column errors for trade-context labels. Supplied outcome columns are exercised separately rather than invented inside a target.

The [Compiled project](examples/compiled-research/README.md#target-variants) runs `target_*.yaml` through the installed CLI. Each class uses a compatible classifier or regressor. Volatility gate outcomes remain explicit. The derived label files contain declared software outcomes from recorded closes, with no future window crossing a chronological partition; they are not venue execution observations. Allocation manifests exercise the native trade-outcome target and bound event context.

## Callable contracts

These entries give the exact public signatures and the source-owned argument, return and failure contracts. Use the project examples above for the research context; a signature is reference evidence, not a claim that every option ran in the base experiment.

### `limen.targets.EmaBreakoutTarget`

<!-- reference: api-limen-targets-emabreakouttarget source=limen/targets/ema_breakout.py region=signature:EmaBreakoutTarget role=signature -->
```python
class EmaBreakoutTarget:
    ...
```

Binary target flagging forward price breakouts above an EMA threshold.

### `limen.targets.EmaBreakoutTarget.__init__`

<!-- reference: api-limen-targets-emabreakouttarget---init-- source=limen/targets/ema_breakout.py region=signature:EmaBreakoutTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:
    ...
```

Store the target column name; no fitting is performed.

| Argument | Contract |
| --- | --- |
| train_data (`pl.DataFrame`) | Training split (not used; kept for interface consistency) |
| target_name (`str`) | Name of the target column to create |

### `limen.targets.EmaBreakoutTarget.transform`

<!-- reference: api-limen-targets-emabreakouttarget-transform source=limen/targets/ema_breakout.py region=signature:EmaBreakoutTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame, target_col: str='close', ema_span: int=30, breakout_delta: float=0.2, breakout_horizon: int=3) -> pl.DataFrame:
    ...
```

Label 1 when the price `breakout_horizon` bars ahead exceeds the EMA

of `target_col` by at least `breakout_delta` of the EMA value.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Split to transform (must have `target_col` column) |
| target_col (`str`) | Column name to analyze for breakouts |
| ema_span (`int`) | Period for EMA calculation |
| breakout_delta (`float`) | Threshold for breakout detection |
| breakout_horizon (`int`) | Forward window in bars |

**Returns:** pl.DataFrame: Data with target column added

### `limen.targets.ExitQualityTarget`

<!-- reference: api-limen-targets-exitqualitytarget source=limen/targets/exit_quality.py region=signature:ExitQualityTarget role=signature -->
```python
class ExitQualityTarget:
    source_columns: ClassVar[list[str]] = list(_REQUIRED_COLUMNS)
```

Categorical target scoring trade exits by reason and net return.

### `limen.targets.ExitQualityTarget.__init__`

<!-- reference: api-limen-targets-exitqualitytarget---init-- source=limen/targets/exit_quality.py region=signature:ExitQualityTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:
    ...
```

Store the target column name; no fitting is performed.

| Argument | Contract |
| --- | --- |
| train_data (`pl.DataFrame`) | Training split (not used; kept for interface consistency) |
| target_name (`str`) | Name of the target column to create |

### `limen.targets.ExitQualityTarget.transform`

<!-- reference: api-limen-targets-exitqualitytarget-transform source=limen/targets/exit_quality.py region=signature:ExitQualityTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame, exit_quality_high: float=1.0, exit_quality_low: float=0.2, exit_quality_medium: float=0.5) -> pl.DataFrame:
    ...
```

Score each closed trade as high, low, or medium quality based on its

`exit_reason` and `exit_net_return`.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Split to transform (must have `exit_reason`, `exit_net_return` columns) |
| exit_quality_high (`float`) | Score for profitable target hits or trailing-stop exits |
| exit_quality_low (`float`) | Score for stop-loss or unprofitable timeout exits |
| exit_quality_medium (`float`) | Score for neutral exits |

**Returns:** pl.DataFrame: Data with target column added

**Raises:** ValueError: If the required user-supplied columns are absent

### `limen.targets.ForwardBreakoutTarget`

<!-- reference: api-limen-targets-forwardbreakouttarget source=limen/targets/forward_breakout.py region=signature:ForwardBreakoutTarget role=signature -->
```python
class ForwardBreakoutTarget:
    ...
```

Binary target for forward price breakouts above a percentage threshold.

### `limen.targets.ForwardBreakoutTarget.__init__`

<!-- reference: api-limen-targets-forwardbreakouttarget---init-- source=limen/targets/forward_breakout.py region=signature:ForwardBreakoutTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:
    ...
```

Store the target column name; no fitting is performed.

| Argument | Contract |
| --- | --- |
| train_data (`pl.DataFrame`) | Training split (not used; kept for interface consistency) |
| target_name (`str`) | Name of the target column to create |

### `limen.targets.ForwardBreakoutTarget.transform`

<!-- reference: api-limen-targets-forwardbreakouttarget-transform source=limen/targets/forward_breakout.py region=signature:ForwardBreakoutTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame, forward_periods: int=24, threshold: float=0.02, shift: int=-1, *, horizon_jitter: int=0, threshold_noise_std: float=0.0, noise_seed: int=0, label_smoothing: float=0.0, permutation_id: str | None=None, training: bool=False) -> pl.DataFrame:
    ...
```

Label 1 if price rises >= threshold over next forward_periods bars.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Split to transform (must have 'close' column) |
| forward_periods (`int`) | Look-ahead window in bars |
| threshold (`float`) | Minimum return for positive label (0.02 = 2%) |
| shift (`int`) | Additional shift applied after labeling |

**Returns:** pl.DataFrame: Data with target column added

### `limen.targets.ForwardVolNormalizedReturnTarget`

<!-- reference: api-limen-targets-forwardvolnormalizedreturntarget source=limen/targets/forward_vol_normalized_return.py region=signature:ForwardVolNormalizedReturnTarget role=signature -->
```python
class ForwardVolNormalizedReturnTarget(VolNormalizedReturnTarget):
    ...
```

Forward log return normalized by current Parkinson volatility.

### `limen.targets.ForwardVolNormalizedReturnTarget.__init__`

<!-- reference: api-limen-targets-forwardvolnormalizedreturntarget---init-- source=limen/targets/forward_vol_normalized_return.py region=signature:ForwardVolNormalizedReturnTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str, periods: int=1, absolute: bool=False, halflife: int=50, min_periods: int=150, high_col: str='high', low_col: str='low', open_col: str='open', close_col: str='close') -> None:
    ...
```

Store target settings and validate Parkinson volatility on train.

| Argument | Contract |
| --- | --- |
| train_data (`pl.DataFrame`) | Training split used for the volatility sanity gate |
| target_name (`str`) | Name of the target column to create |
| periods (`int`) | Forward return horizon in bars |
| absolute (`bool`) | If True, use absolute forward log return |
| halflife (`int`) | EWMA half-life in bars |
| min_periods (`int`) | Warmup before volatility values are emitted |
| high_col (`str`) | High price column |
| low_col (`str`) | Low price column |
| open_col (`str`) | Open price column used for dirty-bar filtering |
| close_col (`str`) | Close price column |

### `limen.targets.ForwardVolNormalizedReturnTarget.transform`

<!-- reference: api-limen-targets-forwardvolnormalizedreturntarget-transform source=limen/targets/forward_vol_normalized_return.py region=signature:ForwardVolNormalizedReturnTarget.transform role=signature -->
```python
@override
def transform(self, data: pl.DataFrame) -> pl.DataFrame:
    ...
```

Compute `log(close.shift(-periods) / close) / parkinson_sigma`.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Split to transform; must have open/high/low/close columns |

**Returns:** pl.DataFrame: Cleaned data with target column added

### `limen.targets.IdentityTarget`

<!-- reference: api-limen-targets-identitytarget source=limen/targets/identity.py region=signature:IdentityTarget role=signature -->
```python
class IdentityTarget:
    ...
```

Target class for columns already present in the data.

### `limen.targets.IdentityTarget.__init__`

<!-- reference: api-limen-targets-identitytarget---init-- source=limen/targets/identity.py region=signature:IdentityTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:
    ...
```

Validate the target column exists; no fitting is performed.

| Argument | Contract |
| --- | --- |
| train_data (`pl.DataFrame`) | Training split; used to validate the target column exists |
| target_name (`str`) | Name of an existing target column |

### `limen.targets.IdentityTarget.transform`

<!-- reference: api-limen-targets-identitytarget-transform source=limen/targets/identity.py region=signature:IdentityTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame) -> pl.DataFrame:
    ...
```

Return data unchanged after validating the target column is present.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Split to transform |

**Returns:** pl.DataFrame: Data unchanged

### `limen.targets.NextBarDownTarget`

<!-- reference: api-limen-targets-nextbardowntarget source=limen/targets/next_bar_down.py region=signature:NextBarDownTarget role=signature -->
```python
class NextBarDownTarget:
    ...
```

Binary target for whether the next close is below the current close.

### `limen.targets.NextBarDownTarget.__init__`

<!-- reference: api-limen-targets-nextbardowntarget---init-- source=limen/targets/next_bar_down.py region=signature:NextBarDownTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:
    ...
```

Store the target column name; no fitting is performed.

| Argument | Contract |
| --- | --- |
| train_data (`pl.DataFrame`) | Training split (not used; kept for interface consistency) |
| target_name (`str`) | Name of the target column to create |

### `limen.targets.NextBarDownTarget.transform`

<!-- reference: api-limen-targets-nextbardowntarget-transform source=limen/targets/next_bar_down.py region=signature:NextBarDownTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame) -> pl.DataFrame:
    ...
```

Label 1 if the next close is lower than the current close.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Split to transform (must have 'close' column) |

**Returns:** pl.DataFrame: Data with target column added

### `limen.targets.NextBarUpTarget`

<!-- reference: api-limen-targets-nextbaruptarget source=limen/targets/next_bar_up.py region=signature:NextBarUpTarget role=signature -->
```python
class NextBarUpTarget:
    ...
```

Binary target for whether the next close is above the current close.

### `limen.targets.NextBarUpTarget.__init__`

<!-- reference: api-limen-targets-nextbaruptarget---init-- source=limen/targets/next_bar_up.py region=signature:NextBarUpTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:
    ...
```

Store the target column name; no fitting is performed.

| Argument | Contract |
| --- | --- |
| train_data (`pl.DataFrame`) | Training split (not used; kept for interface consistency) |
| target_name (`str`) | Name of the target column to create |

### `limen.targets.NextBarUpTarget.transform`

<!-- reference: api-limen-targets-nextbaruptarget-transform source=limen/targets/next_bar_up.py region=signature:NextBarUpTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame) -> pl.DataFrame:
    ...
```

Label 1 if the next close is higher than the current close.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Split to transform (must have 'close' column) |

**Returns:** pl.DataFrame: Data with target column added

### `limen.targets.NextReturnTarget`

<!-- reference: api-limen-targets-nextreturntarget source=limen/targets/next_return.py region=signature:NextReturnTarget role=signature -->
```python
class NextReturnTarget:
    ...
```

Continuous target as next-period percentage return on close price.

### `limen.targets.NextReturnTarget.__init__`

<!-- reference: api-limen-targets-nextreturntarget---init-- source=limen/targets/next_return.py region=signature:NextReturnTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:
    ...
```

Store the target column name; no fitting is performed.

| Argument | Contract |
| --- | --- |
| train_data (`pl.DataFrame`) | Training split (not used; kept for interface consistency) |
| target_name (`str`) | Name of the target column to create |

### `limen.targets.NextReturnTarget.transform`

<!-- reference: api-limen-targets-nextreturntarget-transform source=limen/targets/next_return.py region=signature:NextReturnTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame, periods: int=1, scale: float=100.0) -> pl.DataFrame:
    ...
```

Compute percentage return over the next N periods.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Split to transform (must have 'close' column) |
| periods (`int`) | Look-ahead window in bars |
| scale (`float`) | Multiplier applied to the raw return (100 = percentage points) |

**Returns:** pl.DataFrame: Data with target column added

### `limen.targets.OutcomeLabels`

<!-- reference: api-limen-targets-outcomelabels source=limen/targets/trade_outcome.py region=signature:OutcomeLabels role=signature -->
```python
@dataclass(frozen=True)
class OutcomeLabels:
    rows: pl.DataFrame
    contract_digest: str
```

### `limen.targets.QuantileBinaryTarget`

<!-- reference: api-limen-targets-quantilebinarytarget source=limen/targets/quantile_binary.py region=signature:QuantileBinaryTarget role=signature -->
```python
class QuantileBinaryTarget:
    ...
```

Binary target label from training-set quantile cutoff.

### `limen.targets.QuantileBinaryTarget.__init__`

<!-- reference: api-limen-targets-quantilebinarytarget---init-- source=limen/targets/quantile_binary.py region=signature:QuantileBinaryTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str, source_column: str, quantile: float) -> None:
    ...
```

Fit quantile cutoff on training data.

| Argument | Contract |
| --- | --- |
| train_data (`pl.DataFrame`) | Training split to compute cutoff from |
| target_name (`str`) | Name of the target column to create |
| source_column (`str`) | Column used to compute the quantile threshold |
| quantile (`float`) | Top-N fraction for positive label (0.3 = top 30%) |

### `limen.targets.QuantileBinaryTarget.transform`

<!-- reference: api-limen-targets-quantilebinarytarget-transform source=limen/targets/quantile_binary.py region=signature:QuantileBinaryTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame, shift: int=0) -> pl.DataFrame:
    ...
```

Apply binary flag and optional temporal shift.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Split to transform |
| shift (`int`) | Periods to shift the label (negative = forward-looking) |

**Returns:** pl.DataFrame: Data with target column added

### `limen.targets.RandomBinaryTarget`

<!-- reference: api-limen-targets-randombinarytarget source=limen/targets/random_binary.py region=signature:RandomBinaryTarget role=signature -->
```python
class RandomBinaryTarget:
    ...
```

Uniformly random binary target, used as a noise benchmark.

### `limen.targets.RandomBinaryTarget.__init__`

<!-- reference: api-limen-targets-randombinarytarget---init-- source=limen/targets/random_binary.py region=signature:RandomBinaryTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str, random_state: int | None=None) -> None:
    ...
```

Store the target column name; no fitting is performed.

| Argument | Contract |
| --- | --- |
| train_data (`pl.DataFrame`) | Training split (not used; kept for interface consistency) |
| target_name (`str`) | Name of the target column to create |
| random_state (`int \| None`) | Seed for the random number generator. When set, labels are reproducible across runs given the same data sizes and call order |

### `limen.targets.RandomBinaryTarget.transform`

<!-- reference: api-limen-targets-randombinarytarget-transform source=limen/targets/random_binary.py region=signature:RandomBinaryTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame) -> pl.DataFrame:
    ...
```

Apply random binary labels.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Split to transform |

**Returns:** pl.DataFrame: Data with target column added

### `limen.targets.RiskRewardRatioTarget`

<!-- reference: api-limen-targets-riskrewardratiotarget source=limen/targets/risk_reward_ratio.py region=signature:RiskRewardRatioTarget role=signature -->
```python
class RiskRewardRatioTarget:
    source_columns: ClassVar[list[str]] = list(_REQUIRED_COLUMNS)
```

Continuous target as the ratio of capturable breakout to absolute drawdown.

### `limen.targets.RiskRewardRatioTarget.__init__`

<!-- reference: api-limen-targets-riskrewardratiotarget---init-- source=limen/targets/risk_reward_ratio.py region=signature:RiskRewardRatioTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str) -> None:
    ...
```

Store the target column name; no fitting is performed.

| Argument | Contract |
| --- | --- |
| train_data (`pl.DataFrame`) | Training split (not used; kept for interface consistency) |
| target_name (`str`) | Name of the target column to create |

### `limen.targets.RiskRewardRatioTarget.transform`

<!-- reference: api-limen-targets-riskrewardratiotarget-transform source=limen/targets/risk_reward_ratio.py region=signature:RiskRewardRatioTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame) -> pl.DataFrame:
    ...
```

Compute `capturable_breakout / (\|max_drawdown\| + EPSILON)` per row.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Split to transform (must have `capturable_breakout`, `max_drawdown` columns) |

**Returns:** pl.DataFrame: Data with target column added

**Raises:** ValueError: If the required user-supplied columns are absent

### `limen.targets.ThresholdBinaryTarget`

<!-- reference: api-limen-targets-thresholdbinarytarget source=limen/targets/threshold_binary.py region=signature:ThresholdBinaryTarget role=signature -->
```python
class ThresholdBinaryTarget:
    ...
```

Binary target from a fixed threshold comparison on a source column.

### `limen.targets.ThresholdBinaryTarget.__init__`

<!-- reference: api-limen-targets-thresholdbinarytarget---init-- source=limen/targets/threshold_binary.py region=signature:ThresholdBinaryTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str, source_column: str, threshold: float) -> None:
    ...
```

Store the fixed threshold; no fitting is performed.

| Argument | Contract |
| --- | --- |
| train_data (`pl.DataFrame`) | Training split (not used; kept for interface consistency) |
| target_name (`str`) | Name of the target column to create |
| source_column (`str`) | Column to compare against threshold |
| threshold (`float`) | Fixed comparison value; rows above this are labeled 1 |

### `limen.targets.ThresholdBinaryTarget.transform`

<!-- reference: api-limen-targets-thresholdbinarytarget-transform source=limen/targets/threshold_binary.py region=signature:ThresholdBinaryTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame, shift: int=-1) -> pl.DataFrame:
    ...
```

Apply binary flag and optional temporal shift.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Split to transform |
| shift (`int`) | Periods to shift the label (negative = forward-looking) |

**Returns:** pl.DataFrame: Data with target column added

### `limen.targets.TradeOutcomeTarget`

<!-- reference: api-limen-targets-tradeoutcometarget source=limen/targets/trade_outcome.py region=signature:TradeOutcomeTarget role=signature -->
```python
class TradeOutcomeTarget:
    requires_trade_context: ClassVar[bool] = True
```

### `limen.targets.TradeOutcomeTarget.__init__`

<!-- reference: api-limen-targets-tradeoutcometarget---init-- source=limen/targets/trade_outcome.py region=signature:TradeOutcomeTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str, *, trade_context: TradeTargetContext, side: Literal['long', 'short']='long', output: Literal['binary', 'net_return']='binary', candidate_exposure: float=1.0) -> None:
    ...
```

### `limen.targets.TradeOutcomeTarget.outcomes`

<!-- reference: api-limen-targets-tradeoutcometarget-outcomes source=limen/targets/trade_outcome.py region=signature:TradeOutcomeTarget.outcomes role=signature -->
```python
@property
def outcomes(self) -> OutcomeLabels:
    ...
```

### `limen.targets.TradeOutcomeTarget.transform`

<!-- reference: api-limen-targets-tradeoutcometarget-transform source=limen/targets/trade_outcome.py region=signature:TradeOutcomeTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame, *, trade_context: TradeTargetContext, **transform_params: object) -> pl.DataFrame:
    ...
```

### `limen.targets.TradeTargetContext`

<!-- reference: api-limen-targets-tradetargetcontext source=limen/targets/trade_outcome.py region=signature:TradeTargetContext role=signature -->
```python
@dataclass(frozen=True)
class TradeTargetContext:
    policy: TradePolicy
    inputs: TradeInputs
    partition_start_ns: int
    partition_end_ns: int
    contract_digest: str
```

### `limen.targets.TradelineLongBinaryTarget`

<!-- reference: api-limen-targets-tradelinelongbinarytarget source=limen/targets/tradeline_long_binary.py region=signature:TradelineLongBinaryTarget role=signature -->
```python
class TradelineLongBinaryTarget:
    ...
```

Binary long target from a train-fitted line-height percentile threshold.

### `limen.targets.TradelineLongBinaryTarget.__init__`

<!-- reference: api-limen-targets-tradelinelongbinarytarget---init-- source=limen/targets/tradeline_long_binary.py region=signature:TradelineLongBinaryTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str, max_duration_hours: int, min_height_pct: float, long_threshold_percentile: float) -> None:
    ...
```

Fit the breakout threshold from long lines detected on the training split.

| Argument | Contract |
| --- | --- |
| train_data (`pl.DataFrame`) | Training split with a 'close' column |
| target_name (`str`) | Name of the target column to create |
| max_duration_hours (`int`) | Exclusive upper bound on line duration in bars |
| min_height_pct (`float`) | Minimum absolute line height as a fraction of start price |
| long_threshold_percentile (`float`) | Percentile of long-line heights used as the breakout threshold, in [0, 100] |

**Raises:** ValueError: If 'close' is missing, the percentile is outside [0, 100],

or no long lines exist in the training split

### `limen.targets.TradelineLongBinaryTarget.transform`

<!-- reference: api-limen-targets-tradelinelongbinarytarget-transform source=limen/targets/tradeline_long_binary.py region=signature:TradelineLongBinaryTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame, lookahead_hours: int=DEFAULT_LOOKAHEAD_HOURS, confirmation_hours: int=DEFAULT_CONFIRMATION_HOURS) -> pl.DataFrame:
    ...
```

Apply the confirmed-breakout label using the fitted threshold.

Label 1 where the maximum close over [t, t + lookahead_hours] reaches

the threshold AND the point return at +confirmation_hours or at

+lookahead_hours also exceeds it; 0 otherwise. Rows without a full

forward window become null.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Split to transform (must have 'close' column) |
| lookahead_hours (`int`) | Forward window for the breakout check |
| confirmation_hours (`int`) | Forward offset for the point-confirmation check |

**Returns:** pl.DataFrame: Data with the target column added

**Raises:** ValueError: If 'close' is missing, a window is below 1, or

confirmation_hours exceeds lookahead_hours

### `limen.targets.TripleBarrierTarget`

<!-- reference: api-limen-targets-triplebarriertarget source=limen/targets/triple_barrier.py region=signature:TripleBarrierTarget role=signature -->
```python
class TripleBarrierTarget:
    ...
```

Ternary label from Lopez de Prado's triple-barrier method.

### `limen.targets.TripleBarrierTarget.__init__`

<!-- reference: api-limen-targets-triplebarriertarget---init-- source=limen/targets/triple_barrier.py region=signature:TripleBarrierTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str, upper_multiple: float=1.0, lower_multiple: float=1.0, max_horizon: int=10, span: int=100, min_periods: int=100, close_col: str='close') -> None:
    ...
```

Store barrier settings and validate them; no fitting is performed.

| Argument | Contract |
| --- | --- |
| train_data (`pl.DataFrame`) | Training split (not used; kept for interface consistency) |
| target_name (`str`) | Name of the target column to create |
| upper_multiple (`float`) | Profit-taking barrier width in volatility units |
| lower_multiple (`float`) | Stop-loss barrier width in volatility units |
| max_horizon (`int`) | Vertical barrier as a forward window in bars |
| span (`int`) | EWMA span for the close-to-close return volatility estimate |
| min_periods (`int`) | Warmup before a volatility value is emitted |
| close_col (`str`) | Close price column used for the barrier path |

### `limen.targets.TripleBarrierTarget.transform`

<!-- reference: api-limen-targets-triplebarriertarget-transform source=limen/targets/triple_barrier.py region=signature:TripleBarrierTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame, *, horizon_jitter: int=0, barrier_noise_std: float=0.0, noise_seed: int=0, permutation_id: str | None=None, training: bool=False) -> pl.DataFrame:
    ...
```

Label each bar by the first barrier its forward close path touches.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Split to transform; must have the close column |

**Returns:** pl.DataFrame: Data with a ternary target column added (1 upper, -1 lower,

0 vertical; null during volatility warmup and when the vertical barrier

extends past the available data)

### `limen.targets.VolNormalizedReturnTarget`

<!-- reference: api-limen-targets-volnormalizedreturntarget source=limen/targets/vol_normalized_return.py region=signature:VolNormalizedReturnTarget role=signature -->
```python
class VolNormalizedReturnTarget:
    ...
```

Continuous target as log return normalized by prior Parkinson volatility.

### `limen.targets.VolNormalizedReturnTarget.__init__`

<!-- reference: api-limen-targets-volnormalizedreturntarget---init-- source=limen/targets/vol_normalized_return.py region=signature:VolNormalizedReturnTarget.__init__ role=signature -->
```python
def __init__(self, train_data: pl.DataFrame, target_name: str, high_col: str='high', low_col: str='low', open_col: str='open', close_col: str='close', halflife: int=50, min_periods: int=150) -> None:
    ...
```

Store target settings and validate Parkinson against close-to-close volatility on train.

| Argument | Contract |
| --- | --- |
| train_data (`pl.DataFrame`) | Training split used for the volatility sanity gate |
| target_name (`str`) | Name of the target column to create |
| high_col (`str`) | High price column |
| low_col (`str`) | Low price column |
| open_col (`str`) | Open price column used for dirty-bar filtering |
| close_col (`str`) | Close price column |
| halflife (`int`) | EWMA half-life in bars |
| min_periods (`int`) | Warmup before volatility values are emitted |

### `limen.targets.VolNormalizedReturnTarget.transform`

<!-- reference: api-limen-targets-volnormalizedreturntarget-transform source=limen/targets/vol_normalized_return.py region=signature:VolNormalizedReturnTarget.transform role=signature -->
```python
def transform(self, data: pl.DataFrame) -> pl.DataFrame:
    ...
```

Compute `log(close / close.shift(1)) / parkinson_sigma.shift(1)`.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Split to transform; must have open/high/low/close columns |

**Returns:** pl.DataFrame: Cleaned data with target column added
