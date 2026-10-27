# Features

Features sit one layer above indicators. They combine, re-express, lag, or contextualize price and volume information into model-ready signals or regime flags.

This page covers feature-helper selection, input requirements, and added columns. For lower-level signal primitives, see [Indicators](Indicators.md).

## Conventions

- Feature helpers append columns and return the original frame with the extra outputs attached.
- Helpers either operate on plain kline data or require trade-derived columns such as `maker_ratio`, `no_of_trades`, `price`, or `quantity`.
- Not every helper is intended as a final predictor. Some are utilities used to build targets or wider feature families.
- Regime helpers output compact categorical-style columns such as `regime_ma_slope` or `regime_price_band`.
- `find_min_d` returns a scalar value instead of a frame; it searches for a stationarity-preserving fractional-differentiation order.

## Configuration constraints

Features and indicators validate configuration before computing their output, including on empty input. Periods and windows require integers; Boolean values and floats (including integer-valued floats) are rejected. Periods are never silently truncated. Rolling windows are positive, with existing TA-compatible bounds retained. Causal shifts may be zero: lag helpers, momentum periods, `micro_momentum`, `window_return`, `price_change_pct`, and Ichimoku displacement preserve that behavior. Negative shifts are rejected.

Real coefficients must be finite, Boolean options require `bool`, and column names must be non-empty strings. Closed choices such as `price_vs_band_regime.band` reject unknown values. Period/hour sequences validate every element; calendar hours are `0..23`. Missing input columns still raise native Polars schema errors.

Clip lower bounds cannot exceed their upper bounds. Breakout percentiles and momentum blend weights are in `[0,1]`; percentile and return thresholds are ordered. Fractional differentiation requires non-negative orders and positive truncation thresholds/steps. `breakout_features.lookback` is at least two because its ROC compares adjacent lags. Existing function-specific restrictions remain in force; there is no general requirement that a fast window precede a slow window.

## A feature in each project

The [Python project](examples/python-research/README.md) computes decimal momentum in caller-owned preparation. The same expression runs before fitting and during inference. Changing the expression changes the meaning of the model's input; retain that equality when extending it.

<!-- example: features-python-momentum source=docs/examples/python-research/research.py region=momentum_feature scenario=selection interface=python role=executed -->
```python
def features(bars, window):
    return bars.with_columns(
        (pl.col("close") / pl.col("close").shift(window) - 1).alias("momentum")
    )
```

The [Compiled project](examples/compiled-research/README.md) declares percent-valued `roc` indicators, then adds the lags and fractional feature below in its `search` preparation case. These projects use different units and models; their scores are not a controlled comparison.

<!-- example: features-compiled-momentum source=docs/examples/compiled-research/research.yaml region=features scenario=search interface=compiled role=executed -->
```yaml
    features:
      - func: limen.features.lag_range
        params:
          col: roc_1
          start: 1
          end: 3
          group: momentum
      - func: limen.features.fractional_diff
        params:
          d: 0.4
          cols: [close]
          threshold: 0.05
          group: price
```

The Python excerpt belongs to selection; the YAML excerpt belongs to search and preparation, with the dependencies and recorded inputs described in each project. The helper tables below describe alternatives; a listed helper is not a claim that the selection run invokes it. Trade-derived helpers require the columns named in their notes.

## Kline-derived position and volatility features

These helpers work directly on bar data and require only kline-style inputs.

| Function | Adds by default | Notes |
|---|---|---|
| `atr_percent_sma` | `atr_percent_sma` | ATR scaled by close price using SMA smoothing. |
| `atr_sma` | `atr_sma` | SMA-smoothed ATR variant. |
| `close_position` | `close_position` | Close location inside the current bar's high-low range. Accepts `window`; `1` preserves single-bar behavior. |
| `close_position_rolling` | `close_position_rolling` | Rolling mean of close location inside the high-low range. |
| `close_ma_distance_atr` | `close_ma_distance_atr` | Close-to-SMA distance normalized by SMA-smoothed true range. |
| `distance_from_ma` | `distance_from_ma` | Close distance from its rolling moving average. |
| `distance_from_high` | `distance_from_high` | Distance from a rolling high. |
| `distance_from_low` | `distance_from_low` | Distance from a rolling low. |
| `gap_high` | `gap_high` | Current high relative to the previous close. |
| `kaufman_efficiency_ratio` | `kaufman_efficiency_ratio` | Directional displacement divided by the rolling path length. |
| `narrow_range` | `narrow_range` | Current range divided by trailing maximum range. |
| `price_range_position` | `price_range_position` | Rolling range position over a wider window. |
| `range_pct` | `range_pct` | Current bar range as a percentage. |
| `stochastic_k_abs` | `stochastic_k_abs` | Absolute distance between stochastic %K and 0.5. |
| `trend_strength` | `trend_strength` | Fast-versus-slow trend strength summary. |
| `volume_to_range` | `volume_to_range` | Rolling mean volume per unit of high-low range. |
| `volume_regime` | `volume_regime` | Volume context over a lookback window. |
| `vwap` | `vwap` | Requires a datetime-like `datetime` column because VWAP resets by trading day. |
| `wick_proportion` | `wick_proportion` | Rolling mean wick share of full candle range. |

## Calendar and cyclical time features

These helpers derive time-of-bar context from `datetime` without depending on earlier indicator columns.

| Function | Adds by default | Notes |
|---|---|---|
| `calendar_time_features` | `hour`, `minute`, `weekday`, `day_of_month`, `day_of_year`, `week_of_year`, `month`, `quarter`, `half_of_year`, `is_weekend` | Adds discrete calendar fields for downstream splits, filters, and rules. `weekday` uses ISO numbering (`Monday=1` to `Sunday=7`), and `week_of_year` uses ISO week numbering. |
| `cyclical_time_features` | `hour_sin`, `hour_cos`, `minute_sin`, `minute_cos`, `weekday_sin`, `weekday_cos`, `day_of_month_sin`, `day_of_month_cos`, `day_of_year_sin`, `day_of_year_cos`, `week_of_year_sin`, `week_of_year_cos`, `month_sin`, `month_cos`, `quarter_sin`, `quarter_cos` | Encodes cyclical calendar fields without introducing artificial ordinal jumps. Uses the same ISO conventions as `calendar_time_features`; weekday cycles are phase-aligned with `weekday - 1` before applying sine/cosine. |
| `is_funding_hour` | `is_funding_hour` | Parameterized funding-cadence hour indicator; default hours are `0`, `8`, and `16`. |
| `time_to_funding` | `hours_to_funding` | Continuous hours until the next funding settlement; default cadence is every `8` hours from `0` UTC, and zero at a settlement bar. |
| `is_us_open_hour` | `is_us_open_hour` | Parameterized US open-hour indicator; default hour is `14`. |

## Range-based volatility features

These helpers estimate volatility directly from OHLC structure instead of relying only on close-to-close returns.

| Function | Adds by default | Notes |
|---|---|---|
| `parkinson_volatility` | `parkinson_volatility` | High-low range estimator that ignores close-to-close drift. |
| `parkinson_vol_of_vol` | `parkinson_vol_of_vol` | Rolling standard deviation of Parkinson variance. |
| `garman_klass_volatility` | `garman_klass_volatility` | OHLC estimator using open, high, low, and close information. |
| `rogers_satchell_volatility` | `rogers_satchell_volatility` | Drift-robust OHLC volatility estimator. |
| `volatility_ratio` | `volatility_ratio` | Short rolling Parkinson variance divided by long rolling Parkinson variance. |
| `volatility_spike` | `volatility_spike` | Current Parkinson variance divided by its fixed-lag value. |
| `yang_zhang_volatility` | `yang_zhang_volatility` | Combines overnight, open-close, and range information into a higher-fidelity volatility estimate. Requires `window > 1`. |

## Liquidity and impact features

These helpers translate ordinary OHLCV bars into liquidity, impact, and slippage proxies.

| Function | Adds by default | Notes |
|---|---|---|
| `dollar_volume` | `dollar_volume` | Price-times-volume activity proxy using close and volume. |
| `amihud_illiquidity` | `amihud_illiquidity` | Absolute return per dollar of volume, a compact price-impact proxy. |
| `volume_ratio` | `volume_ratio` | Volume relative to its simple-moving-average baseline. |
| `return_per_dollar_volume` | `return_per_dollar_volume` | Signed return per dollar of volume for directional impact analysis. |
| `range_per_dollar_volume` | `range_per_dollar_volume` | Bar range scaled by dollar volume. |
| `illiquidity_shock` | `illiquidity_shock` | Current Amihud-style illiquidity relative to its own trailing mean. |
| `liquidity_drop` | `liquidity_drop` | Current LOB liquidity divided by fixed-lag LOB liquidity. |
| `liquidity_range` | `liquidity_range` | Rolling mean high-liquidity to low-liquidity ratio. |
| `maker_liquidity_share` | `maker_liquidity_share` | Maker liquidity divided by total liquidity. |
| `maker_volume_share` | `maker_volume_share` | Maker volume divided by total volume. |
| `maker_volume_ratio` | `maker_volume_ratio` | Rolling mean maker-volume share. |
| `taker_imbalance_ratio` | `taker_imbalance_ratio` | Rolling mean absolute taker imbalance as a share of volume. |
| `trade_density` | `trade_density` | Rolling mean number of trades per unit of volume. |
| `trade_imbalance` | `trade_imbalance` | Rolling maker volume divided by rolling total volume. |
| `trade_size_ratio` | `trade_size_ratio` | Short average trade size divided by long average trade size. |
| `bulk_volume_classification` | `bvc_buy_volume`, `bvc_sell_volume` | Splits bar volume into buy and sell by bulk-volume classification (standardized return through the normal CDF). Needs only `close` and `volume`. |
| `order_flow_imbalance` | `order_flow_imbalance` plus `bvc_buy_volume`, `bvc_sell_volume` | Rolling net BVC-classified signed flow as a share of volume; a bar-level order-flow imbalance proxy, not level-2 OFI. |
| `vpin` | `vpin` plus `bvc_buy_volume`, `bvc_sell_volume` | Volume-synchronized Probability of Informed Trading from the BVC buy/sell imbalance; a flow-toxicity gauge. Feed volume bars for canonical equal-volume buckets. |

### Dollar-bar crash reversal

`dollar_bar_crash_reversal` is the primary signal used by the bundled rule-based SFD of the same name. It requires a UTC-sorted `datetime` plus `open`, `close`, `liquidity_sum`, and `maker_liquidity`, and appends the Int8 column `dollar_bar_crash_reversal_position`.

For row `t`, the transform:

1. finds the latest `open` at or before `t - 4h` by backward as-of join and computes `log(close_t / reference_open) * 10_000`
2. computes maker flow as `1 - 2 * maker_liquidity / liquidity_sum` only where liquidity is finite and positive
3. standardizes flow against causal 30-day rolling medians of flow and absolute deviation, both closed on the left with at least 100 observations
4. triggers when momentum is at or below `momentum_threshold_bps` and the robust flow score is above `flow_z_threshold`
5. holds the trigger active for `hold_minutes` of wall-clock time

The structural core permits a new trigger only when the next row belongs to the same UTC date. Therefore the last row of each UTC day, including the final row in the input, cannot initiate a trigger. A position initiated earlier may remain active there until its wall-clock hold expires. This is a **one-row availability** boundary: the exact research trigger is not same-row causal. The built-in backtest's one-bar execution lag is an execution adaptation, not a claim that the raw trigger was knowable on row `t`.

The hold is time-based rather than bar-count-based. Dollar bars arrive irregularly, so the physical span represented by a 60-minute hold can exceed 60 minutes between observed execution rows.

## Realized risk and tail features

These helpers describe the quality of recent movement, not just its level.

| Function | Adds by default | Notes |
|---|---|---|
| `realized_semivariance` | `upside_semivariance`, `downside_semivariance` | Splits rolling squared returns into upside and downside components. |
| `downside_volatility_ratio` | `downside_volatility_ratio` | Rolling downside squared-return share of total squared returns. |
| `realized_skewness` | `realized_skewness` | Rolling skewness of close-to-close returns. |
| `realized_kurtosis` | `realized_kurtosis` | Rolling kurtosis of close-to-close returns. |
| `jump_variation_proxy` | `jump_variation_proxy` | Positive gap between realized variance and bipower variation proxy. |
| `tail_event_intensity` | `tail_event_intensity` | Share of recent bars whose absolute return exceeds a configurable threshold. |
| `volatility_of_volatility` | `volatility_of_volatility` | Rolling variability of rolling close-to-close return volatility. |
| `return_autocorrelation` | `return_autocorrelation` | Rolling correlation between returns and one-bar lagged returns. |
| `return_volatility_correlation` | `return_volatility_correlation` | Rolling correlation between returns and Parkinson variance. |
| `volume_volatility_correlation` | `volume_volatility_correlation` | Rolling correlation between volume and Parkinson variance. |

## Seasonality-normalized features

These helpers compare current bar behavior to the trailing mean for the same hour of the week.

Custom SFD authors must keep seasonality baselines causal. Split-wide normalization such as `.mean().over(['season_weekday', 'season_hour'])`, or any full-split hour-of-week baseline computed in one pass, leaks future information within the split. Use trailing-only or train-fitted seasonality baselines instead. The built-in `relative_volume_seasonality`, `relative_range_seasonality`, and `relative_volatility_seasonality` helpers use trailing hour-of-week baselines.

| Function | Adds by default | Notes |
|---|---|---|
| `relative_volume_seasonality` | `relative_volume_seasonality` | Current volume relative to the trailing baseline for the same hour-of-week bucket. |
| `relative_range_seasonality` | `relative_range_seasonality` | Current range percentage relative to the trailing hour-of-week baseline. |
| `relative_volatility_seasonality` | `relative_volatility_seasonality` | Current absolute return magnitude relative to the trailing hour-of-week baseline. |

## Candle structure and auction features

These helpers focus on how a bar moved internally, not just where it finished.

| Function | Adds by default | Notes |
|---|---|---|
| `body_to_range` | `body_to_range` | Absolute candle body size divided by the full bar range. |
| `wick_imbalance` | `wick_imbalance` | Upper-wick minus lower-wick imbalance as a share of full range. |
| `range_overlap` | `range_overlap` | Overlap share between the current bar range and the previous bar range. |
| `rejection_intensity` | `rejection_intensity` | Wick-heavy rejection proxy based on total wick share and directional close location. |
| `absorption_intensity` | `absorption_intensity` | High-volume, small-body absorption proxy using a trailing shifted volume baseline. |

## Cross-timescale context features

These helpers summarize cross-horizon agreement or disagreement on market state.

| Function | Adds by default | Notes |
|---|---|---|
| `trend_coherence` | `trend_coherence` | Average sign agreement across short, medium, and long return horizons. |
| `volatility_term_structure` | `volatility_term_structure` | Average ratio between short, medium, and long rolling volatility estimates. |
| `sma_ratios` | `<price>_sma_<period>` and `sma_<period>_ratio` per configured period | Price-to-SMA ratios across multiple horizons, keeping the SMA columns. |

## Breakout and regime features

These helpers provide state or structure rather than only a continuous numeric series.

| Function | Adds by default | Notes |
|---|---|---|
| `breakout_features` | lagged breakout columns plus `long_roll_mean`, `long_roll_std`, `short_roll_mean`, `short_roll_std`, `roc_long_12_1`, `roc_short_12_1` | Enriches pre-existing breakout flags. |
| `breakout_percentile_regime` | `price_range_position`, `regime_breakout_pct` | Uses percentile thresholds over price-range position. |
| `hh_hl_structure_regime` | `regime_hh_hl` | Captures higher-high and higher-low style structure. |
| `ichimoku_cloud` | `tenkan`, `kijun`, `senkou_a`, `senkou_b`, `chikou` | Full Ichimoku feature set. |
| `ma_slope_regime` | `regime_ma_slope` | Regime label based on moving-average slope. |
| `price_vs_band_regime` | `regime_price_band` | Uses price distance relative to a band definition. |
| `sma_crossover` | `crossover`, `signal` | Compact crossover-state helper. |
| `window_return_regime` | `ret_24`, `regime_window_return` | Return plus regime thresholding over a window. |

## Line-based context features

These helpers summarize how recently and how densely price interacted with detected price lines — pairs of bars at most `max_duration_hours` apart whose close-to-close change is at least `min_height_pct` (positive lines are long, negative are short; quantile lines are those at or above the `quantile_threshold` height quantile per direction).

The two grouped transforms below are the YAML-facing surface. Each detects lines internally from scalar params on the frame it receives — per split under the manifest pipeline, so detection cannot observe other splits — and adds its full column family in one detection pass. The composites use array geometry from `limen.utils.find_price_lines`; dictionary utilities remain available independently.

| Function | Adds by default | Notes |
|---|---|---|
| `price_lines` | `active_lines`, `hours_since_big_move`, `line_momentum_<m>h`, `trending_score`, `reversal_potential` | All-line family: span count, end recency (capped at `big_move_lookback_hours`), and long-minus-short end counts over the trailing `[t-m, t)` window with their balance (`trending_score`, in `[-1, 1]`) and min/max ratio (`reversal_potential`, in `[0, 1]`). |
| `quantile_price_lines` | `hours_since_quantile_line`, `active_quantile_count`, `quantile_line_density_<d>h`, `quantile_momentum_<m>h`, `avg_quantile_height_<h>h`, `quantile_direction_bias` | Quantile-line family: end recency and span count, end density over `density_lookback_hours`, signed height sum over ends in `[t-m, t]`, and mean height plus height-weighted direction (in `[-1, 1]`) over ends in `[t-h, t]`. |

`active_lines` and `active_quantile_count` count lines that span the current bar before the line's end — the event that defines it — is knowable. That within-line lookahead is inherited from the tradeline research design: treat both columns as research-only, not live-computable. The grouped transforms expose `include_research_only`; set it to `false` to omit the active-span columns. Bundled live-safe templates set `include_research_only: false`. The end-event columns are causal. For `quantile_price_lines`, each row uses the per-direction height quantile of lines completed through that row, including ties. `active_quantile_count` retains its retrospective whole-frame selection.

The per-column building blocks below take pre-computed line structures (`list[dict]` with `start_idx`/`end_idx`) and remain available for programmatic composition.

| Function | Adds by default | Notes |
|---|---|---|
| `active_lines` | `active_lines` | Count of long and short lines active at each bar. |
| `active_quantile_count` | `active_quantile_count` | Active-line count restricted to quantile-filtered lines. |
| `quantile_line_density` | `quantile_line_density_<lookback>h` | Count of quantile-line endings within a trailing `lookback_hours` window. |
| `hours_since_big_move` | `hours_since_big_move` | Bars since the most recent line end, capped at `lookback_hours`. |
| `hours_since_quantile_line` | `hours_since_quantile_line` | Bars since the most recent quantile-line end, capped at `lookback_hours`. |

### Line calculation performance

The composite transforms keep line geometry in arrays; the standalone dictionary helpers retain their existing API. Baseline comparisons cover eager/lazy frames, research/live outputs, quantile endpoints, short histories and missing values. Quantile end features match the frozen implementation on each tested prefix's final row; their latency check compares equivalent as-of work.

| Feature | Rows | Live wall ms, before → after | Live CPU ms, before → after | Peak RSS MiB, before → after |
|---|---:|---:|---:|---:|
| `price_lines` | 20,000 | 649.1 → 62.8 | 548.5 → 49.1 | 501 → 309 |
| `price_lines` | 83,136 | 2692.9 → 236.7 | 2424.9 → 230.0 | 1360 → 498 |

These are local M1 Max measurements on real BTCUSDT bars, Python 3.12.12, NumPy 2.5.1 and Polars 1.42.1, with numerical-library thread limits set to one. Values are medians of three warm calls; first-call samples, research-mode results, input digests and exact schemas are in the [raw measurements](https://github.com/Vaquum/Limen/blob/main/tests/fixtures/line_performance_benchmark.json). Peak RSS includes imports and input loading. The raw archive also retains historical whole-frame quantile timings; those do not measure the corrected as-of calculation. Desktop contention inflated some wall samples, particularly the quantile baseline; CPU timings provide the corresponding execution cost.

## Lag helpers and threshold utilities

These helpers expand existing columns or define cutoffs for target construction.

| Function | Adds or returns | Notes |
|---|---|---|
| `lag_column` | one lagged column such as `close_lag_2` | Requires `col` and `lag`. |
| `lag_columns` | one lag per listed column | Requires `cols` and `lag`. |
| `lag_range` | a lag range such as `close_lag_1` through `close_lag_3` | Requires `col`, `start`, and `end`. |
| `lag_range_cols` | a lag range for each listed column | Requires `cols`, `start`, and `end`. |
| `rolling_zscore` | configurable `*_zscore_*` column | Applies `identity`, `log1p`, or `abs` before rolling z-score standardization. |
| `cusum_filter` | `cusum_event` | Int8 flag of symmetric CUSUM events on the close log-return path (`1` up, `-1` down, `0` none); gates which moves are worth sampling. |

## Stationarity and long-memory helpers

These helpers reduce non-stationarity while preserving more long-memory structure than a first difference would.

| Function | Adds or returns | Notes |
|---|---|---|
| `fractional_diff` | one `*_fracdiff` column per selected input column | Applies fixed-width fractional differentiation. Original columns are preserved. |
| `find_min_d` | scalar `d` value | Iterates over candidate orders and uses the Augmented Dickey-Fuller test to find the smallest stationary order. |

Two practical details matter:

- `fractional_diff` needs `cols=['close']` and writes new columns such as `close_fracdiff`.
- if one split is too short to produce the same fractional-diff column as another split, Manifest now drops that extra column during split alignment so the final `data_dict` stays consistent.

## Trade-shape and microstructure features

These helpers need richer data than ordinary OHLCV bars.

| Function | Adds by default | Notes |
|---|---|---|
| `kline_imbalance` | `imbalance` | Requires `maker_ratio` and `no_of_trades` from data retrieval or bar formation. |
| `conserved_flux_renormalization` | synthetic OHLCV plus `value_sum`, `vwap`, `flux_rel_std_mean`, `flux_rel_std_var`, `entropy_mean`, `entropy_var`, `Δflux_rms`, `Δentropy_rms` | Works on trade-level `datetime`, `price`, and `quantity`, then rolls those into kline-aligned diagnostics. |

## Dynamic-target and entry-score features

This family builds volatility-conditioned targets, stops, and regime weights, plus the microstructure entry score they combine with. The helpers compose: `volatility_measure` and `regime_multiplier` feed `dynamic_target` and `dynamic_stop_loss`, the momentum and candle-position helpers feed `entry_score_microstructure`, and `feature_aliases` snapshots the family into `*_feature` columns with nulls filled.

| Function | Adds by default | Notes |
|---|---|---|
| `close_to_extremes` | `close_to_high`, `close_to_low` | Close position relative to bar high and low extremes. |
| `dynamic_stop_loss` | `dynamic_stop_loss` | Volatility- and regime-conditioned stop-loss level. |
| `dynamic_target` | `dynamic_target` | Volatility- and regime-conditioned target level. |
| `ema_alignment` | `ema`, `ema_alignment` | EMA alignment score with power transformation. |
| `entry_score_microstructure` | `entry_score`, `entry_score_base` | Microstructure timing score from momentum, spread, candle position, and volume spikes. |
| `feature_aliases` | `dynamic_target_feature`, `entry_score_feature`, `momentum_score_feature`, `regime_high_feature`, `regime_low_feature`, `regime_normal_feature`, `vol_60h_feature`, `vol_percentile_feature` | Null-filled aliases snapshotting the family for model consumption. |
| `log_returns` | `log_returns` | Logarithmic returns of the close series. |
| `market_regime` | `sma_20`, `sma_50`, `trend_strength`, `volatility_ratio`, `sma_48`, `volume_sma`, `volume_regime`, `market_favorable` | Trend-strength and volume-regime favorability score. |
| `micro_momentum` | `micro_momentum` | Short-horizon price momentum. |
| `momentum_confirmation` | `momentum_score` | Momentum confirmation score from recent price changes. |
| `momentum_periods` | `momentum_<period>` per configured period | Momentum over multiple horizons. |
| `momentum_weight` | `momentum_weight` | Momentum-direction weighting factor. |
| `position_in_candle` | `position_in_candle` | Close position within the bar high-low range. |
| `position_in_range` | `position_in_range` | Close position within the bar high-low range over a rolling window. |
| `regime_multiplier` | `regime_multiplier` | Volatility-regime multiplier for dynamic parameter adjustment. |
| `returns_lags` | `returns_lag_<lag>` per configured lag | Lagged simple returns. |
| `spread` | `spread` | High-low range normalized by close (same formula as `range_pct`, but stored in a `spread` column). |
| `spread_percent` | `spread_percent` | High-low range normalized by close, stored as `spread_percent` for microstructure scoring. |
| `volatility_1h` | `volatility_1h` | Alias of an existing volatility column at the one-hour horizon. |
| `volatility_measure` | `volatility_measure` | Combined rolling-volatility and ATR-percentage measure. |
| `volatility_weight` | `volatility`, `volatility_weight` | Inverse-volatility weighting factor. |
| `volume_spike` | `volume_spike` | Volume relative to a rolling-statistics baseline. |
| `volume_trend` | `volume_trend` | Short-term versus long-term volume average trend. |

`market_regime` binds `sma_20` and `sma_50` to the selected short and long close averages, even for equal or swapped periods. `volume_sma` is the selected volume average. Fixed aliases take precedence over conflicting close `sma_<period>` names; close outputs take precedence over the volume `sma_<lookback>` output. Nonconflicting parameterized outputs remain in their existing order. Incoming `sma_20`, `sma_50` or `volume_sma` columns are rejected, including repeated application. These bindings apply to this call; later transforms retain their own output behavior. SMA warmup uses NaN, while the derived regime features use null.

## Choosing between indicators and features

- Use an indicator for a direct market calculation such as RSI, ATR, or MACD.
- Use a feature for structure around those signals, such as lags, regimes, relative position, or multi-step aggregation.
- Use the lag helpers when the main value is temporal context rather than a new market calculation.
- Use `fractional_diff` when stationarity itself is part of the design problem, not just a preprocessing afterthought.

## Read next

- [Indicators](Indicators.md) for lower-level signal primitives
- [Transforms](Transforms.md) for stateless target shaping and cleanup helpers
- [Calibration](Calibration.md) for post-model probability calibration and threshold optimization
- [Experiment Manifest](Experiment-Manifest.md) for how features plug into the split-first manifest pipeline
- [Utilities](Utilities.md) for the exported `adf_test()` helper that `find_min_d` builds on

## Private volatility barriers

`limen.features.volatility_barriers.volatility_barriers` is a private backtest producer, not an ordinary primary-feature transform; see [Backtest](Backtest.md#causal-entry-barriers-research-only) for configuration and evidence restrictions.

For simple close return `r[t] = close[t]/close[t-1]-1`, calculate EWMA standard deviations of `r[t-1]` using spans 24 and 168 by default (`adjust=True`, `bias=False`, 24 observations minimum). Require integer `2 <= min_periods <= span <= regime_span`. The first `min_periods+1` rows are warmup. Select regime multiplier 0.8 below fast/slow ratio 0.75, 1.2 above 1.25 and 1 otherwise, including equality and both-zero volatility. TP/SL are `10000 * fast_std * regime_multiplier * side_multiple`, clipped to shared minimum/maximum (10/2000 bps). Default TP/SL multiples are 2/1. All coefficients are finite positive; clips are ordered and below 10000 bps. No future return or partition-local EWMA restart enters the calculation.

## Callable contracts

These entries give the exact public signatures and the source-owned argument, return and failure contracts. Use the project examples above for the research context; a signature is reference evidence, not a claim that every option ran in the base experiment.

### `limen.features.absorption_intensity`

<!-- reference: api-limen-features-absorption-intensity source=limen/features/absorption_intensity.py region=signature:absorption_intensity role=signature -->
```python
def absorption_intensity(data: pl.DataFrame, window: int=20, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close', volume_col: str='volume') -> pl.DataFrame:
    ...
```

Compute high-effort low-result absorption intensity from volume and candle body share.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with open, high, low, close, and volume columns |
| window (`int`) | Number of periods used for the trailing volume baseline |
| open_col (`str`) | Column name for open prices |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |
| volume_col (`str`) | Column name for traded volume |

**Returns:** pl.DataFrame: The input data with a new column 'absorption_intensity'

### `limen.features.active_lines`

<!-- reference: api-limen-features-active-lines source=limen/features/active_lines.py region=signature:active_lines role=signature -->
```python
def active_lines(data: pl.DataFrame, long_lines: list[dict[str, float]], short_lines: list[dict[str, float]]) -> pl.DataFrame:
    ...
```

Compute active line count per row from provided long and short line spans.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'datetime' and 'close' columns |
| long_lines (`list[dict]`) | Long line definitions with 'start_idx' and 'end_idx' |
| short_lines (`list[dict]`) | Short line definitions with 'start_idx' and 'end_idx' |

**Returns:** pl.DataFrame: The input data with a new column 'active_lines'

### `limen.features.active_quantile_count`

<!-- reference: api-limen-features-active-quantile-count source=limen/features/active_quantile_count.py region=signature:active_quantile_count role=signature -->
```python
def active_quantile_count(data: pl.DataFrame, long_lines_q: list[dict[str, float]], short_lines_q: list[dict[str, float]]) -> pl.DataFrame:
    ...
```

Compute count of active quantile-filtered lines per row.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'datetime' and 'close' columns |
| long_lines_q (`list[dict]`) | Quantile-filtered long lines with 'start_idx' and 'end_idx' |
| short_lines_q (`list[dict]`) | Quantile-filtered short lines with 'start_idx' and 'end_idx' |

**Returns:** pl.DataFrame: The input data with a new column 'active_quantile_count'

### `limen.features.amihud_illiquidity`

<!-- reference: api-limen-features-amihud-illiquidity source=limen/features/amihud_illiquidity.py region=signature:amihud_illiquidity role=signature -->
```python
def amihud_illiquidity(data: pl.DataFrame, price_col: str='close', volume_col: str='volume') -> pl.DataFrame:
    ...
```

Compute the Amihud illiquidity proxy from price changes and dollar volume.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with price and volume columns |
| price_col (`str`) | Column name used for close-to-close returns |
| volume_col (`str`) | Column name for traded volume |

**Returns:** pl.DataFrame: The input data with a new column 'amihud_illiquidity'

### `limen.features.atr_percent_sma`

<!-- reference: api-limen-features-atr-percent-sma source=limen/features/atr_percent_sma.py region=signature:atr_percent_sma role=signature -->
```python
def atr_percent_sma(data: pl.DataFrame, period: int=14) -> pl.DataFrame:
    ...
```

Compute ATR as percentage of close price using Simple Moving Average.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'high', 'low', 'close' columns |
| period (`int`) | Number of periods for ATR calculation |

**Returns:** pl.DataFrame: The input data with a new column 'atr_percent_sma'

### `limen.features.atr_sma`

<!-- reference: api-limen-features-atr-sma source=limen/features/atr_sma.py region=signature:atr_sma role=signature -->
```python
def atr_sma(data: pl.DataFrame, period: int=14) -> pl.DataFrame:
    ...
```

Compute Average True Range using Simple Moving Average.

NOTE: Different from standard ATR which uses Wilder's smoothing.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'high', 'low', 'close' columns |
| period (`int`) | Number of periods for ATR calculation |

**Returns:** pl.DataFrame: The input data with a new column 'atr_sma'

### `limen.features.body_to_range`

<!-- reference: api-limen-features-body-to-range source=limen/features/body_to_range.py region=signature:body_to_range role=signature -->
```python
def body_to_range(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

Compute candle body size relative to the full bar range.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with open, high, low, and close columns |
| open_col (`str`) | Column name for open prices |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |

**Returns:** pl.DataFrame: The input data with a new column 'body_to_range'

### `limen.features.breakout_features`

<!-- reference: api-limen-features-breakout-features source=limen/features/breakout_features.py region=signature:breakout_features role=signature -->
```python
def breakout_features(data: pl.DataFrame, long_col: str='breakout_long', short_col: str='breakout_short', lookback: int=12, horizon: int=12, target: str='breakout_pct') -> pl.DataFrame:
    ...
```

Compute comprehensive breakout-related features including lags, stats, and ROC.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with breakout signal columns |
| long_col (`str`) | Column name for long breakout signals |
| short_col (`str`) | Column name for short breakout signals |
| lookback (`int`) | Number of periods for feature calculation |
| horizon (`int`) | Number of periods to shift for avoiding lookahead bias |
| target (`str`) | Target column name for filtering null values |

**Returns:** pl.DataFrame: The input data with multiple breakout feature columns added

### `limen.features.breakout_percentile_regime`

<!-- reference: api-limen-features-breakout-percentile-regime source=limen/features/breakout_percentile_regime.py region=signature:breakout_percentile_regime role=signature -->
```python
def breakout_percentile_regime(df: pl.DataFrame, period: int=24, p_hi: float=0.85, p_lo: float=0.15) -> pl.DataFrame:
    ...
```

Compute regime classification by percentile position of 'close' within rolling [low, high].

| Argument | Contract |
| --- | --- |
| df (`pl.DataFrame`) | Klines dataset with 'high', 'low', 'close' columns |
| period (`int`) | Rolling window for high/low range |
| p_hi (`float`) | Upper percentile threshold in [0, 1] |
| p_lo (`float`) | Lower percentile threshold in [0, 1] |

**Returns:** pl.DataFrame: The input data with a new column 'regime_breakout_pct'

### `limen.features.bulk_volume_classification`

<!-- reference: api-limen-features-bulk-volume-classification source=limen/features/bulk_volume_classification.py region=signature:bulk_volume_classification role=signature -->
```python
def bulk_volume_classification(data: pl.DataFrame, window: int=20, close_col: str='close', volume_col: str='volume') -> pl.DataFrame:
    ...
```

Split each bar's volume into buy and sell volume by bulk-volume classification.

The close-to-close log return is standardized by its rolling standard

deviation and passed through the standard normal CDF to estimate the buy

fraction (Easley, Lopez de Prado, and O'Hara). Buy volume is that fraction

times total volume; sell volume is the remainder. Direction is inferred

from price action alone, so no trade-side data is required.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with close and volume columns |
| window (`int`) | Rolling window for the return-volatility standardization |
| close_col (`str`) | Column name used for close-to-close log returns |
| volume_col (`str`) | Column name for total traded volume |

**Returns:** pl.DataFrame: The input data with new columns 'bvc_buy_volume' and 'bvc_sell_volume'

### `limen.features.calendar_time_features`

<!-- reference: api-limen-features-calendar-time-features source=limen/features/calendar_time_features.py region=signature:calendar_time_features role=signature -->
```python
def calendar_time_features(df: pl.DataFrame) -> pl.DataFrame:
    ...
```

Add discrete calendar time features from the datetime column.

| Argument | Contract |
| --- | --- |
| df (`pl.DataFrame`) | Input frame with a datetime column named `datetime` |

**Returns:** pl.DataFrame: The input frame with `hour`, `minute`, `weekday`,

`day_of_month`, `day_of_year`, `week_of_year`, `month`,

`quarter`, `half_of_year`, and `is_weekend` appended. `weekday` uses ISO

numbering with `Monday=1` through `Sunday=7`, and

`week_of_year` follows ISO week numbering

### `limen.features.close_ma_distance_atr`

<!-- reference: api-limen-features-close-ma-distance-atr source=limen/features/close_ma_distance_atr.py region=signature:close_ma_distance_atr role=signature -->
```python
def close_ma_distance_atr(data: pl.DataFrame, ma_window: int=20, atr_window: int=14, high_col: str='high', low_col: str='low', close_col: str='close', output_col: str='close_ma_distance_atr') -> pl.DataFrame:
    ...
```

Compute close-to-SMA distance normalized by SMA-smoothed true range.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with high, low, and close columns |
| ma_window (`int`) | Number of periods in the simple moving average |
| atr_window (`int`) | Number of periods in the true-range moving average |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the ATR-normalized distance column appended

### `limen.features.close_position`

<!-- reference: api-limen-features-close-position source=limen/features/close_position.py region=signature:close_position role=signature -->
```python
def close_position(data: pl.DataFrame, window: int=1, high_col: str='high', low_col: str='low', close_col: str='close', output_col: str='close_position') -> pl.DataFrame:
    ...
```

Compute close position within the high-low range as a unit interval fraction.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with high, low, and close columns |
| window (`int`) | Rolling mean window; `1` preserves the single-bar behavior |
| high_col (`str`) | High price column |
| low_col (`str`) | Low price column |
| close_col (`str`) | Close price column |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with a new close-position column

### `limen.features.close_position_rolling`

<!-- reference: api-limen-features-close-position-rolling source=limen/features/close_position_rolling.py region=signature:close_position_rolling role=signature -->
```python
def close_position_rolling(data: pl.DataFrame, window: int=20, high_col: str='high', low_col: str='low', close_col: str='close', output_col: str='close_position_rolling') -> pl.DataFrame:
    ...
```

Compute rolling mean close position inside the high-low range.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with high, low, and close columns |
| window (`int`) | Number of periods in the rolling mean |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the rolling close-position column appended

### `limen.features.close_to_extremes`

<!-- reference: api-limen-features-close-to-extremes source=limen/features/close_to_extremes.py region=signature:close_to_extremes role=signature -->
```python
def close_to_extremes(data: pl.DataFrame) -> pl.DataFrame:
    ...
```

Compute close position relative to high and low extremes.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'close', 'high', 'low' columns |

**Returns:** pl.DataFrame: The input data with new columns 'close_to_high' and 'close_to_low'

### `limen.features.conserved_flux_renormalization`

<!-- reference: api-limen-features-conserved-flux-renormalization source=limen/features/conserved_flux_renormalization.py region=signature:conserved_flux_renormalization role=signature -->
```python
def conserved_flux_renormalization(trades_df: pl.DataFrame, *, kline_interval: str='1h', base_window_s: int=60, levels: int=6) -> pl.DataFrame:
    ...
```

Compute multi-scale, conserved-flux features and their deviation scores

for each k-line—turning raw trade ticks into a six-value fingerprint that

flags hours where the dollar flow or trade-size entropy breaks

scale-invariant behaviour.

| Argument | Contract |
| --- | --- |
| trades_df (`pl.DataFrame`) | The trades DataFrame. |
| kline_interval (`str`) | The kline interval. |
| base_window_s (`int`) | The base window size. |
| levels (`int`) | The number of levels to compute. |

**Returns:** pl.DataFrame: A klines DataFrame with the CFR features

### `limen.features.cusum_filter`

<!-- reference: api-limen-features-cusum-filter source=limen/features/cusum_filter.py region=signature:cusum_filter role=signature -->
```python
def cusum_filter(data: pl.DataFrame, threshold: float=0.01, close_col: str='close', output_col: str='cusum_event') -> pl.DataFrame:
    ...
```

Flag symmetric CUSUM events on the close-price log-return path.

Running sums accumulate positive and negative log returns and reset when

either breaches `threshold`, emitting an event at that bar (Lopez de Prado's

symmetric CUSUM filter). The filter samples only moves whose cumulative

magnitude is meaningful, gating out micro-noise ahead of downstream sampling.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame \| pl.LazyFrame`) | Klines dataset with a close price column |
| threshold (`float`) | Cumulative log-return magnitude that triggers an event |
| close_col (`str`) | Column name used for close-to-close log returns |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame \| pl.LazyFrame: The input data, matching the input frame

type, with an Int8 column: 1 for an up event, -1 for a down event,

0 otherwise

NOTE: Column access is expression-based so the feature runs on both eager

DataFrames and the LazyFrames piped by the manifest feature pipeline.

### `limen.features.cyclical_time_features`

<!-- reference: api-limen-features-cyclical-time-features source=limen/features/cyclical_time_features.py region=signature:cyclical_time_features role=signature -->
```python
def cyclical_time_features(df: pl.DataFrame) -> pl.DataFrame:
    ...
```

Add sine and cosine encodings for cyclical datetime fields.

| Argument | Contract |
| --- | --- |
| df (`pl.DataFrame`) | Input frame with a datetime column named `datetime` |

**Returns:** pl.DataFrame: The input frame with cyclical encodings for hour,

minute, weekday, day_of_month, day_of_year, week_of_year, month,

and quarter appended as `*_sin` and `*_cos` columns. Weekday and

week-of-year use ISO numbering, and weekday cycles are phase-

aligned with `weekday - 1` before sine/cosine encoding

### `limen.features.distance_from_high`

<!-- reference: api-limen-features-distance-from-high source=limen/features/distance_from_high.py region=signature:distance_from_high role=signature -->
```python
def distance_from_high(data: pl.DataFrame, period: int=24) -> pl.DataFrame:
    ...
```

Compute distance from rolling high as percentage.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'high', 'close' columns |
| period (`int`) | Number of periods for rolling high calculation |

**Returns:** pl.DataFrame: The input data with a new column 'distance_from_high'

### `limen.features.distance_from_low`

<!-- reference: api-limen-features-distance-from-low source=limen/features/distance_from_low.py region=signature:distance_from_low role=signature -->
```python
def distance_from_low(data: pl.DataFrame, period: int=24) -> pl.DataFrame:
    ...
```

Compute distance from rolling low as percentage.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'low', 'close' columns |
| period (`int`) | Number of periods for rolling low calculation |

**Returns:** pl.DataFrame: The input data with a new column 'distance_from_low'

### `limen.features.distance_from_ma`

<!-- reference: api-limen-features-distance-from-ma source=limen/features/distance_from_ma.py region=signature:distance_from_ma role=signature -->
```python
def distance_from_ma(data: pl.DataFrame, window: int=20, close_col: str='close', output_col: str='distance_from_ma') -> pl.DataFrame:
    ...
```

Compute close distance from its rolling moving average.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with a close price column |
| window (`int`) | Number of periods in the moving average |
| close_col (`str`) | Column name for close prices |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the moving-average distance column appended

### `limen.features.dollar_bar_crash_reversal`

<!-- reference: api-limen-features-dollar-bar-crash-reversal source=limen/features/dollar_bar_crash_reversal.py region=signature:dollar_bar_crash_reversal role=signature -->
```python
def dollar_bar_crash_reversal(data: pl.DataFrame | pl.LazyFrame, momentum_threshold_bps: float, flow_z_threshold: float, hold_minutes: int, *, bar_interval_seconds: int | None=None) -> pl.DataFrame | pl.LazyFrame:
    ...
```

Build prior-only momentum/robust-flow crash-reversal positions.

Hold triggers for the wall-clock window; a verified interval removes the next-row day-mask dependency.

### `limen.features.dollar_volume`

<!-- reference: api-limen-features-dollar-volume source=limen/features/dollar_volume.py region=signature:dollar_volume role=signature -->
```python
def dollar_volume(data: pl.DataFrame, price_col: str='close', volume_col: str='volume') -> pl.DataFrame:
    ...
```

Compute traded dollar volume from price and volume columns.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with price and volume columns |
| price_col (`str`) | Column name used as the price proxy for traded value |
| volume_col (`str`) | Column name for traded volume |

**Returns:** pl.DataFrame: The input data with a new column 'dollar_volume'

### `limen.features.downside_volatility_ratio`

<!-- reference: api-limen-features-downside-volatility-ratio source=limen/features/downside_volatility_ratio.py region=signature:downside_volatility_ratio role=signature -->
```python
def downside_volatility_ratio(data: pl.DataFrame, window: int=20, close_col: str='close', output_col: str='downside_volatility_ratio') -> pl.DataFrame:
    ...
```

Compute rolling downside squared-return share of total squared returns.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with a close price column |
| window (`int`) | Number of periods in the rolling sums |
| close_col (`str`) | Column name for close prices |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the downside-volatility ratio appended

### `limen.features.dynamic_stop_loss`

<!-- reference: api-limen-features-dynamic-stop-loss source=limen/features/dynamic_stop_loss.py region=signature:dynamic_stop_loss role=signature -->
```python
def dynamic_stop_loss(data: pl.DataFrame, base_stop_loss: float, stop_volatility_multiplier: float, clip_lower_mult: float=0.7, clip_upper_mult: float=1.4) -> pl.DataFrame:
    ...
```

Compute dynamic stop loss levels based on volatility conditions and regime.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'volatility_measure', 'regime_multiplier' columns |
| base_stop_loss (`float`) | Base stop loss threshold |
| stop_volatility_multiplier (`float`) | Multiplier for volatility-adjusted stops |
| clip_lower_mult (`float`) | Lower bound multiplier for clipping |
| clip_upper_mult (`float`) | Upper bound multiplier for clipping |

**Returns:** pl.DataFrame: The input data with a new column 'dynamic_stop_loss'

### `limen.features.dynamic_target`

<!-- reference: api-limen-features-dynamic-target source=limen/features/dynamic_target.py region=signature:dynamic_target role=signature -->
```python
def dynamic_target(data: pl.DataFrame, base_min_breakout: float, target_volatility_multiplier: float, clip_lower_mult: float=0.6, clip_upper_mult: float=1.4) -> pl.DataFrame:
    ...
```

Compute dynamic target levels based on volatility conditions and regime.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'volatility_measure', 'regime_multiplier' columns |
| base_min_breakout (`float`) | Base minimum breakout threshold |
| target_volatility_multiplier (`float`) | Multiplier for volatility-adjusted targets |
| clip_lower_mult (`float`) | Lower bound multiplier for clipping |
| clip_upper_mult (`float`) | Upper bound multiplier for clipping |

**Returns:** pl.DataFrame: The input data with a new column 'dynamic_target'

### `limen.features.ema_alignment`

<!-- reference: api-limen-features-ema-alignment source=limen/features/ema_alignment.py region=signature:ema_alignment role=signature -->
```python
def ema_alignment(data: pl.DataFrame, ema_span: int=21, power: float=2.0) -> pl.DataFrame:
    ...
```

Compute EMA alignment score with power transformation.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'close' column |
| ema_span (`int`) | Span for EMA calculation |
| power (`float`) | Power for alignment transformation |

**Returns:** pl.DataFrame: The input data with new columns 'ema', 'ema_alignment'

### `limen.features.entry_score_microstructure`

<!-- reference: api-limen-features-entry-score-microstructure source=limen/features/entry_score_microstructure.py region=signature:entry_score_microstructure role=signature -->
```python
def entry_score_microstructure(data: pl.DataFrame, micro_momentum_period: int=3, volume_spike_period: int=20, spread_mean_period: int=48, entry_position_weight_base: float=0.25, entry_momentum_weight_base: float=0.25, entry_volume_weight_base: float=0.25, entry_spread_weight_base: float=0.25, entry_position_weight_low_vol: float=0.35, entry_momentum_weight_low_vol: float=0.15, entry_volume_weight_low_vol: float=0.15, entry_spread_weight_low_vol: float=0.35, entry_position_weight_high_vol: float=0.15, entry_momentum_weight_high_vol: float=0.35, entry_volume_weight_high_vol: float=0.35, entry_spread_weight_high_vol: float=0.15, entry_volume_spike_min: float=0.5, entry_volume_spike_max: float=1.5, entry_volume_spike_normalizer: float=1.5, entry_spread_ratio_min: float=0, entry_spread_ratio_max: float=2) -> pl.DataFrame:
    ...
```

Compute sophisticated entry score based on microstructure timing signals.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'high', 'low', 'close', 'volume', 'volatility_regime' columns |
| micro_momentum_period (`int`) | Period for micro momentum calculation |
| volume_spike_period (`int`) | Period for volume spike calculation |
| spread_mean_period (`int`) | Period for spread normalization |
| entry_position_weight_base (`float`) | Base weight for position-in-candle factor |
| entry_momentum_weight_base (`float`) | Base weight for momentum factor |
| entry_volume_weight_base (`float`) | Base weight for volume factor |
| entry_spread_weight_base (`float`) | Base weight for spread factor |
| entry_position_weight_low_vol (`float`) | Position weight in low volatility regime |
| entry_momentum_weight_low_vol (`float`) | Momentum weight in low volatility regime |
| entry_volume_weight_low_vol (`float`) | Volume weight in low volatility regime |
| entry_spread_weight_low_vol (`float`) | Spread weight in low volatility regime |
| entry_position_weight_high_vol (`float`) | Position weight in high volatility regime |
| entry_momentum_weight_high_vol (`float`) | Momentum weight in high volatility regime |
| entry_volume_weight_high_vol (`float`) | Volume weight in high volatility regime |
| entry_spread_weight_high_vol (`float`) | Spread weight in high volatility regime |
| entry_volume_spike_min (`float`) | Minimum volume spike value for clipping |
| entry_volume_spike_max (`float`) | Maximum volume spike value for clipping |
| entry_volume_spike_normalizer (`float`) | Normalizer for volume spike scaling |
| entry_spread_ratio_min (`float`) | Minimum spread ratio for clipping |
| entry_spread_ratio_max (`float`) | Maximum spread ratio for clipping |

**Returns:** pl.DataFrame: The input data with a new column 'entry_score'

### `limen.features.feature_aliases`

<!-- reference: api-limen-features-feature-aliases source=limen/features/feature_aliases.py region=signature:feature_aliases role=signature -->
```python
def feature_aliases(df: pl.DataFrame, base_min_breakout: float=0.005, volatility_regime_enabled: bool=True) -> pl.DataFrame:
    ...
```

Compute feature aliases with null filling for dynamic and regime features.

| Argument | Contract |
| --- | --- |
| df (`pl.DataFrame`) | Klines dataset with dynamic features |
| base_min_breakout (`float`) | Default value for dynamic_target |
| volatility_regime_enabled (`bool`) | Whether to create regime feature aliases |

**Returns:** pl.DataFrame: The input data with new columns 'dynamic_target_feature', 'entry_score_feature', 'momentum_score_feature' and optional regime columns

### `limen.features.find_min_d`

<!-- reference: api-limen-features-find-min-d source=limen/features/fractional_diff.py region=signature:find_min_d role=signature -->
```python
def find_min_d(data: pl.DataFrame, col: str, d_start: float=0.0, d_end: float=1.0, step: float=0.1, significance_level: float=0.05, threshold: float=1e-05) -> float:
    ...
```

Find the minimum fractional differentiation order achieving stationarity.

Iterates from d_start to d_end by step, applying fractional

differentiation at each d and testing with ADF. Returns the

smallest d where the series is stationary.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Input data containing the column to test |
| col (`str`) | Column name to differentiate and test |
| d_start (`float`) | Starting differentiation order |
| d_end (`float`) | Maximum differentiation order |
| step (`float`) | Increment between tested d values |
| significance_level (`float`) | ADF test significance threshold |
| threshold (`float`) | Weight truncation threshold for fractional diff |

**Returns:** float: Smallest d achieving stationarity, or d_end if none found

### `limen.features.fractional_diff`

<!-- reference: api-limen-features-fractional-diff source=limen/features/fractional_diff.py region=signature:fractional_diff role=signature -->
```python
def fractional_diff(data: pl.DataFrame, d: float=0.0, cols: list[str] | None=None, threshold: float=1e-05) -> pl.DataFrame:
    ...
```

Apply Fixed-Width Fractional Differentiation (FFD) to specified columns.

Based on Lopez de Prado's AFML Chapter 5. Achieves stationarity

while preserving memory in financial time series. Adds new columns

with suffix '_fracdiff' — original columns are preserved for

indicators that depend on raw values.

NOTE: AFML recommends applying to log-transformed prices for best results.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Input data |
| d (`float`) | Finite, non-negative order (0 = identity copy, 1 = standard diff) |
| cols (`list[str] \| None`) | Columns to differentiate (required) |
| threshold (`float`) | Finite, positive weight truncation threshold per AFML standard |

**Returns:** pl.DataFrame: Data with new '{col}_fracdiff' columns added.

When d=0, the new columns are copies of the originals.

**Raises:** ValueError: If cols is None, empty, a string, or names missing columns,

d is non-finite or negative, threshold is non-finite or nonpositive,

or a recurrence weight is non-finite.

### `limen.features.gap_high`

<!-- reference: api-limen-features-gap-high source=limen/features/gap_high.py region=signature:gap_high role=signature -->
```python
def gap_high(data: pl.DataFrame) -> pl.DataFrame:
    ...
```

Compute gap between current high and previous close as percentage.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'high' and 'close' columns |

**Returns:** pl.DataFrame: The input data with a new column 'gap_high'

### `limen.features.garman_klass_volatility`

<!-- reference: api-limen-features-garman-klass-volatility source=limen/features/garman_klass_volatility.py region=signature:garman_klass_volatility role=signature -->
```python
def garman_klass_volatility(data: pl.DataFrame, window: int=20, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

Compute Garman-Klass range-based volatility over a rolling window.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with open, high, low, and close price columns |
| window (`int`) | Number of periods used for the rolling estimator |
| open_col (`str`) | Column name for open prices |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |

**Returns:** pl.DataFrame: The input data with a new column 'garman_klass_volatility'

### `limen.features.hh_hl_structure_regime`

<!-- reference: api-limen-features-hh-hl-structure-regime source=limen/features/hh_hl_structure_regime.py region=signature:hh_hl_structure_regime role=signature -->
```python
def hh_hl_structure_regime(df: pl.DataFrame, window: int=24, score_threshold: int=4) -> pl.DataFrame:
    ...
```

Compute regime by higher-high / higher-low market structure within a rolling window.

| Argument | Contract |
| --- | --- |
| df (`pl.DataFrame`) | Klines dataset with 'high', 'low' columns |
| window (`int`) | Rolling window size for structure count |
| score_threshold (`int`) | Absolute score threshold for Up/Down classification |

**Returns:** pl.DataFrame: The input data with a new column 'regime_hh_hl'

### `limen.features.hours_since_big_move`

<!-- reference: api-limen-features-hours-since-big-move source=limen/features/hours_since_big_move.py region=signature:hours_since_big_move role=signature -->
```python
def hours_since_big_move(data: pl.DataFrame, long_lines: list[dict[str, float]], short_lines: list[dict[str, float]], lookback_hours: int) -> pl.DataFrame:
    ...
```

Compute hours since the most recent line end capped by lookback_hours.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'datetime' and 'close' columns |
| long_lines (`list[dict]`) | Long line definitions with 'start_idx' and 'end_idx' |
| short_lines (`list[dict]`) | Short line definitions with 'start_idx' and 'end_idx' |
| lookback_hours (`int`) | Maximum number of hours to track recency |

**Returns:** pl.DataFrame: The input data with a new column 'hours_since_big_move'

### `limen.features.hours_since_quantile_line`

<!-- reference: api-limen-features-hours-since-quantile-line source=limen/features/hours_since_quantile_line.py region=signature:hours_since_quantile_line role=signature -->
```python
def hours_since_quantile_line(data: pl.DataFrame, long_lines_q: list[dict[str, float]], short_lines_q: list[dict[str, float]], lookback_hours: int) -> pl.DataFrame:
    ...
```

Compute hours since the most recent quantile-filtered line end capped by lookback_hours.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'datetime' and 'close' columns |
| long_lines_q (`list[dict]`) | Quantile-filtered long lines with 'end_idx' |
| short_lines_q (`list[dict]`) | Quantile-filtered short lines with 'end_idx' |
| lookback_hours (`int`) | Maximum number of hours to track recency |

**Returns:** pl.DataFrame: The input data with a new column 'hours_since_quantile_line'

### `limen.features.ichimoku_cloud`

<!-- reference: api-limen-features-ichimoku-cloud source=limen/features/ichimoku_cloud.py region=signature:ichimoku_cloud role=signature -->
```python
def ichimoku_cloud(data: pl.DataFrame, tenkan_period: int=9, kijun_period: int=26, senkou_b_period: int=52, displacement: int=26) -> pl.DataFrame:
    ...
```

Compute Ichimoku Cloud components for trend and momentum analysis.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'high', 'low', 'close' columns |
| tenkan_period (`int`) | Lookback period for Tenkan-sen |
| kijun_period (`int`) | Lookback period for Kijun-sen |
| senkou_b_period (`int`) | Lookback period for Senkou Span B |
| displacement (`int`) | Non-negative shift for Senkou spans and Chikou span |

**Returns:** pl.DataFrame: The input data with new columns:

'tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou'

### `limen.features.illiquidity_shock`

<!-- reference: api-limen-features-illiquidity-shock source=limen/features/illiquidity_shock.py region=signature:illiquidity_shock role=signature -->
```python
def illiquidity_shock(data: pl.DataFrame, window: int=20, price_col: str='close', volume_col: str='volume') -> pl.DataFrame:
    ...
```

Compute the current Amihud illiquidity level relative to its trailing baseline.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with price and volume columns |
| window (`int`) | Number of periods used for the trailing Amihud baseline |
| price_col (`str`) | Column name used for close-to-close returns |
| volume_col (`str`) | Column name for traded volume |

**Returns:** pl.DataFrame: The input data with a new column 'illiquidity_shock'

### `limen.features.is_funding_hour`

<!-- reference: api-limen-features-is-funding-hour source=limen/features/is_funding_hour.py region=signature:is_funding_hour role=signature -->
```python
def is_funding_hour(data: pl.DataFrame, hours: Sequence[int]=(0, 8, 16), datetime_col: str='datetime', output_col: str='is_funding_hour') -> pl.DataFrame:
    ...
```

Mark rows whose hour matches a configured funding cadence.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with a datetime-like column |
| hours (`Sequence[int]`) | Funding hours to mark |
| datetime_col (`str`) | Column name for timestamps |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with an Int8 funding-hour flag appended

### `limen.features.is_us_open_hour`

<!-- reference: api-limen-features-is-us-open-hour source=limen/features/is_us_open_hour.py region=signature:is_us_open_hour role=signature -->
```python
def is_us_open_hour(data: pl.DataFrame, hour: int=14, datetime_col: str='datetime', output_col: str='is_us_open_hour') -> pl.DataFrame:
    ...
```

Mark rows whose hour matches the configured US open hour.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with a datetime-like column |
| hour (`int`) | Hour to mark |
| datetime_col (`str`) | Column name for timestamps |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with an Int8 US-open-hour flag appended

### `limen.features.jump_variation_proxy`

<!-- reference: api-limen-features-jump-variation-proxy source=limen/features/jump_variation_proxy.py region=signature:jump_variation_proxy role=signature -->
```python
def jump_variation_proxy(data: pl.DataFrame, window: int=20, close_col: str='close') -> pl.DataFrame:
    ...
```

Compute a rolling jump-variation proxy from realized variance and aligned bipower variation.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with a close price column |
| window (`int`) | Number of returns used for the rolling jump proxy |
| close_col (`str`) | Column name used for log returns |

**Returns:** pl.DataFrame: The input data with a new column 'jump_variation_proxy'

### `limen.features.kaufman_efficiency_ratio`

<!-- reference: api-limen-features-kaufman-efficiency-ratio source=limen/features/kaufman_efficiency_ratio.py region=signature:kaufman_efficiency_ratio role=signature -->
```python
def kaufman_efficiency_ratio(data: pl.DataFrame, window: int=10, close_col: str='close', output_col: str='kaufman_efficiency_ratio') -> pl.DataFrame:
    ...
```

Compute Kaufman's efficiency ratio over a rolling window.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with a close price column |
| window (`int`) | Number of periods used for displacement and path length |
| close_col (`str`) | Column name for close prices |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the efficiency-ratio column appended

### `limen.features.kline_imbalance`

<!-- reference: api-limen-features-kline-imbalance source=limen/features/kline_imbalance.py region=signature:kline_imbalance role=signature -->
```python
def kline_imbalance(data: pl.DataFrame, window: int=14) -> pl.DataFrame:
    ...
```

Compute rolling buyer/seller imbalance over klines instead of raw trades.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'maker_ratio' and 'no_of_trades' columns |
| window (`int`) | Number of periods for rolling window calculation |

**Returns:** pl.DataFrame: The input data with a new column 'imbalance'

### `limen.features.lag_column`

<!-- reference: api-limen-features-lag-column source=limen/features/lagged_features.py region=signature:lag_column role=signature -->
```python
def lag_column(data: pl.DataFrame, col: str, lag: int, alias: object=None) -> pl.DataFrame:
    ...
```

Compute a lagged version of a column.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with specified column |
| col (`str`) | The column name to lag |
| lag (`int`) | The number of periods to lag |
| alias (`str, optional`) | New column name. If None, uses alias f"lag_{lag}" |

**Returns:** pl.DataFrame: The input data with the lagged column appended

### `limen.features.lag_columns`

<!-- reference: api-limen-features-lag-columns source=limen/features/lagged_features.py region=signature:lag_columns role=signature -->
```python
def lag_columns(data: pl.DataFrame, cols: list[str], lag: int) -> pl.DataFrame:
    ...
```

Compute lagged versions of multiple columns.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with specified columns |
| cols (`list[str]`) | The list of column names to lag |
| lag (`int`) | The number of periods to lag |

**Returns:** pl.DataFrame: The input data with the lagged columns appended

### `limen.features.lag_range`

<!-- reference: api-limen-features-lag-range source=limen/features/lagged_features.py region=signature:lag_range role=signature -->
```python
def lag_range(data: pl.DataFrame, col: str, start: int, end: int) -> pl.DataFrame:
    ...
```

Compute multiple lagged versions of a column over a range.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with specified column |
| col (`str`) | The column name to lag |
| start (`int`) | The start of lag range (inclusive) |
| end (`int`) | The end of lag range (inclusive) |

**Returns:** pl.DataFrame: The input data with the lagged columns appended

### `limen.features.lag_range_cols`

<!-- reference: api-limen-features-lag-range-cols source=limen/features/lagged_features.py region=signature:lag_range_cols role=signature -->
```python
def lag_range_cols(data: pl.DataFrame, cols: list[str], start: object, end: object) -> pl.DataFrame:
    ...
```

Compute multiple lagged versions of multiple columns over a range.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with specified columns |
| cols (`list[str]`) | The list of column names to lag |
| start (`int`) | The start of lag range (inclusive) |
| end (`int`) | The end of lag range (inclusive) |

**Returns:** pl.DataFrame: The input data with the lagged columns appended

### `limen.features.liquidity_drop`

<!-- reference: api-limen-features-liquidity-drop source=limen/features/liquidity_drop.py region=signature:liquidity_drop role=signature -->
```python
def liquidity_drop(data: pl.DataFrame, window: int=20, liquidity_col: str='liquidity_sum', output_col: str='liquidity_drop') -> pl.DataFrame:
    ...
```

Compute current liquidity relative to liquidity from a fixed lag.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with a total-liquidity column |
| window (`int`) | Lag used as the denominator |
| liquidity_col (`str`) | Column name for total liquidity |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the liquidity-drop column appended

### `limen.features.liquidity_range`

<!-- reference: api-limen-features-liquidity-range source=limen/features/liquidity_range.py region=signature:liquidity_range role=signature -->
```python
def liquidity_range(data: pl.DataFrame, window: int=20, high_liquidity_col: str='high_liquidity', low_liquidity_col: str='low_liquidity', output_col: str='liquidity_range') -> pl.DataFrame:
    ...
```

Compute rolling mean high-liquidity to low-liquidity ratio.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with high- and low-liquidity columns |
| window (`int`) | Number of periods in the rolling mean |
| high_liquidity_col (`str`) | Column name for high-side liquidity |
| low_liquidity_col (`str`) | Column name for low-side liquidity |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the liquidity-range column appended

### `limen.features.log_returns`

<!-- reference: api-limen-features-log-returns source=limen/features/log_returns.py region=signature:log_returns role=signature -->
```python
def log_returns(data: pl.DataFrame, price_col: str='close') -> pl.DataFrame:
    ...
```

Compute logarithmic returns for price series.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with price column |
| price_col (`str`) | Name of the price column (default: 'close') |

**Returns:** pl.DataFrame: The input data with a new column 'log_returns'

### `limen.features.ma_slope_regime`

<!-- reference: api-limen-features-ma-slope-regime source=limen/features/ma_slope_regime.py region=signature:ma_slope_regime role=signature -->
```python
def ma_slope_regime(df: pl.DataFrame, period: int=24, threshold: float=0.0, normalize_by_std: bool=True) -> pl.DataFrame:
    ...
```

Compute regime using the slope of SMA(close, period) with optional normalization.

| Argument | Contract |
| --- | --- |
| df (`pl.DataFrame`) | Klines dataset with 'close' column |
| period (`int`) | SMA period for the slope |
| threshold (`float`) | Slope threshold; applied after normalization when enabled |
| normalize_by_std (`bool`) | Whether to divide slope by rolling std(period) |

**Returns:** pl.DataFrame: The input data with a new column 'regime_ma_slope'

### `limen.features.maker_liquidity_share`

<!-- reference: api-limen-features-maker-liquidity-share source=limen/features/maker_liquidity_share.py region=signature:maker_liquidity_share role=signature -->
```python
def maker_liquidity_share(data: pl.DataFrame, maker_liquidity_col: str='maker_liquidity', liquidity_col: str='liquidity_sum', output_col: str='maker_liquidity_share') -> pl.DataFrame:
    ...
```

Compute maker liquidity as a share of total liquidity.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with maker and total liquidity columns |
| maker_liquidity_col (`str`) | Column name for maker-side liquidity |
| liquidity_col (`str`) | Column name for total liquidity |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the maker-liquidity share column appended

### `limen.features.maker_volume_ratio`

<!-- reference: api-limen-features-maker-volume-ratio source=limen/features/maker_volume_ratio.py region=signature:maker_volume_ratio role=signature -->
```python
def maker_volume_ratio(data: pl.DataFrame, window: int=20, maker_volume_col: str='maker_volume', volume_col: str='volume', output_col: str='maker_volume_ratio') -> pl.DataFrame:
    ...
```

Compute the rolling mean maker-volume share.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with maker volume and total volume columns |
| window (`int`) | Number of periods in the rolling mean |
| maker_volume_col (`str`) | Column name for maker-side traded volume |
| volume_col (`str`) | Column name for total traded volume |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the rolling maker-volume ratio appended

### `limen.features.maker_volume_share`

<!-- reference: api-limen-features-maker-volume-share source=limen/features/maker_volume_share.py region=signature:maker_volume_share role=signature -->
```python
def maker_volume_share(data: pl.DataFrame, maker_volume_col: str='maker_volume', volume_col: str='volume', output_col: str='maker_volume_share') -> pl.DataFrame:
    ...
```

Compute maker volume as a share of total volume.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with maker volume and total volume columns |
| maker_volume_col (`str`) | Column name for maker-side traded volume |
| volume_col (`str`) | Column name for total traded volume |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the maker-volume share column appended

### `limen.features.market_regime`

<!-- reference: api-limen-features-market-regime source=limen/features/market_regime.py region=signature:market_regime role=signature -->
```python
def market_regime(df: pl.DataFrame, lookback: int=48, short_sma: int=20, long_sma: int=50) -> pl.DataFrame:
    ...
```

Add close-average, volume-regime and market-favorability features.

sma_20 and sma_50 bind the selected short/long close averages, including equal

or swapped periods; volume_sma binds the selected volume average. Fixed aliases

take precedence over close canonicals, which take precedence over the volume

canonical. Nonconflicting sma_<period> outputs retain their existing positions.

Incoming fixed aliases raise DuplicateError. SMA warmup remains NaN; derived

trend, volatility, volume-regime and favorability warmup remains null.

### `limen.features.micro_momentum`

<!-- reference: api-limen-features-micro-momentum source=limen/features/micro_momentum.py region=signature:micro_momentum role=signature -->
```python
def micro_momentum(data: pl.DataFrame, period: int=3) -> pl.DataFrame:
    ...
```

Compute short-term price momentum over specified periods.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'close' column |
| period (`int`) | Number of periods for momentum calculation |

**Returns:** pl.DataFrame: The input data with a new column 'micro_momentum'

### `limen.features.momentum_confirmation`

<!-- reference: api-limen-features-momentum-confirmation source=limen/features/momentum_confirmation.py region=signature:momentum_confirmation role=signature -->
```python
def momentum_confirmation(df: pl.DataFrame, short_period: int=1, long_period: int=3, short_weight: float=0.5) -> pl.DataFrame:
    ...
```

Compute simple momentum confirmation scores based on recent price changes.

| Argument | Contract |
| --- | --- |
| df (`pl.DataFrame`) | Klines dataset with 'close' column |
| short_period (`int`) | Number of periods for short-term momentum |
| long_period (`int`) | Number of periods for long-term momentum |
| short_weight (`float`) | Weight for short-term momentum in final score |

**Returns:** pl.DataFrame: The input data with new columns 'momentum_1', 'momentum_3', 'momentum_score'

### `limen.features.momentum_periods`

<!-- reference: api-limen-features-momentum-periods source=limen/features/momentum_periods.py region=signature:momentum_periods role=signature -->
```python
def momentum_periods(data: pl.DataFrame, periods: list[int] | None=None, price_col: str='close') -> pl.DataFrame:
    ...
```

Compute momentum over multiple time periods.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with price column |
| periods (`list`) | Non-negative periods for momentum calculation |
| price_col (`str`) | Name of the price column (default: 'close') |

**Returns:** pl.DataFrame: The input data with new columns 'momentum_{period}' for each period

### `limen.features.momentum_weight`

<!-- reference: api-limen-features-momentum-weight source=limen/features/momentum_weight.py region=signature:momentum_weight role=signature -->
```python
def momentum_weight(data: pl.DataFrame, period: int=12, weight_multiplier: float=0.5, base_weight: float=0.5) -> pl.DataFrame:
    ...
```

Compute momentum-based weighting factor from price change direction.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'close' column |
| period (`int`) | Period for momentum calculation |
| weight_multiplier (`float`) | Multiplier for positive momentum |
| base_weight (`float`) | Base weight added to all observations |

**Returns:** pl.DataFrame: The input data with a new column 'momentum_weight'

### `limen.features.narrow_range`

<!-- reference: api-limen-features-narrow-range source=limen/features/narrow_range.py region=signature:narrow_range role=signature -->
```python
def narrow_range(data: pl.DataFrame, window: int=20, high_col: str='high', low_col: str='low', output_col: str='narrow_range') -> pl.DataFrame:
    ...
```

Compute current range relative to the trailing maximum range.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with high and low price columns |
| window (`int`) | Number of periods in the trailing maximum |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the narrow-range column appended

### `limen.features.order_flow_imbalance`

<!-- reference: api-limen-features-order-flow-imbalance source=limen/features/order_flow_imbalance.py region=signature:order_flow_imbalance role=signature -->
```python
def order_flow_imbalance(data: pl.DataFrame, window: int=20, classification_window: int=20, close_col: str='close', volume_col: str='volume') -> pl.DataFrame:
    ...
```

Compute a rolling bulk-volume order-flow imbalance from classified buy and sell volume.

Bars are split into buy and sell volume by bulk-volume classification, then

the net signed volume is summed over `window` and divided by the summed total

volume, giving a directional flow proxy in [-1, 1]. This is a bar-level proxy

for order-flow imbalance inferred from price; it is not the level-2 order-book

OFI of Cont, Kukanov, and Stoikov.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with close and volume columns |
| window (`int`) | Rolling window over which signed flow is accumulated |
| classification_window (`int`) | Rolling window for the bulk-volume classification standardization |
| close_col (`str`) | Column name used for close-to-close log returns |
| volume_col (`str`) | Column name for total traded volume |

**Returns:** pl.DataFrame: The input data with classified buy/sell volume columns and a

new 'order_flow_imbalance' column

### `limen.features.parkinson_vol_of_vol`

<!-- reference: api-limen-features-parkinson-vol-of-vol source=limen/features/parkinson_vol_of_vol.py region=signature:parkinson_vol_of_vol role=signature -->
```python
def parkinson_vol_of_vol(data: pl.DataFrame, window: int=20, high_col: str='high', low_col: str='low', output_col: str='parkinson_vol_of_vol') -> pl.DataFrame:
    ...
```

Compute rolling standard deviation of Parkinson variance.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with high and low price columns |
| window (`int`) | Number of periods in the rolling standard deviation |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the Parkinson vol-of-vol column appended

### `limen.features.parkinson_volatility`

<!-- reference: api-limen-features-parkinson-volatility source=limen/features/parkinson_volatility.py region=signature:parkinson_volatility role=signature -->
```python
def parkinson_volatility(data: pl.DataFrame, window: int=20, high_col: str='high', low_col: str='low') -> pl.DataFrame:
    ...
```

Compute Parkinson range-based volatility over a rolling window.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with high and low price columns |
| window (`int`) | Number of periods used for the rolling estimator |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |

**Returns:** pl.DataFrame: The input data with a new column 'parkinson_volatility'

### `limen.features.position_in_candle`

<!-- reference: api-limen-features-position-in-candle source=limen/features/position_in_candle.py region=signature:position_in_candle role=signature -->
```python
def position_in_candle(data: pl.DataFrame) -> pl.DataFrame:
    ...
```

Compute position of close within candle high-low range.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'high', 'low', 'close' columns |

**Returns:** pl.DataFrame: The input data with a new column 'position_in_candle'

### `limen.features.position_in_range`

<!-- reference: api-limen-features-position-in-range source=limen/features/position_in_range.py region=signature:position_in_range role=signature -->
```python
def position_in_range(df: pl.DataFrame) -> pl.DataFrame:
    ...
```

Compute position of close within candle high-low range.

Uses the following calculation:

(close - low) / (high - low + 1e-10)

| Argument | Contract |
| --- | --- |
| df (`pl.DataFrame`) | Klines dataset with 'high', 'low', 'close' columns |

**Returns:** pl.DataFrame: The input data with a new column 'position_in_range'

### `limen.features.price_lines`

<!-- reference: api-limen-features-price-lines source=limen/features/price_lines.py region=signature:price_lines role=signature -->
```python
def price_lines(data: pl.DataFrame, max_duration_hours: int, min_height_pct: float, big_move_lookback_hours: int=DEFAULT_BIG_MOVE_LOOKBACK_HOURS, momentum_lookback_hours: int=DEFAULT_MOMENTUM_LOOKBACK_HOURS, include_research_only: object=True) -> pl.DataFrame:
    ...
```

Compute line-based context features from internally detected price lines.

Detects lines on the frame it receives (per split under the manifest

pipeline) and adds five columns: 'active_lines' (count of lines spanning

the bar), 'hours_since_big_move' (bars since the most recent line end,

capped at big_move_lookback_hours), 'line_momentum_<m>h' (long minus

short line ends in the trailing window [t-m, t)), 'trending_score'

(signed end-count balance in [-1, 1]), and 'reversal_potential'

(min/max end-count ratio in [0, 1]).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame \| pl.LazyFrame`) | Klines dataset with a 'close' column |
| max_duration_hours (`int`) | Exclusive upper bound on line duration in bars |
| min_height_pct (`float`) | Minimum absolute line height as a fraction of start price |
| big_move_lookback_hours (`int`) | Recency cap for 'hours_since_big_move' |
| momentum_lookback_hours (`int`) | Trailing window for the end-count columns |
| include_research_only (`bool`) | Include the non-live-computable 'active_lines' span-count column. Set False for live-safe feature surfaces. |

**Returns:** pl.DataFrame \| pl.LazyFrame: The input data, matching the input frame

type, with five new line-based columns

**Raises:** ValueError: If a lookback is below 1

NOTE: Column access is expression-based so the feature runs on both eager

DataFrames and the LazyFrames piped by the manifest feature pipeline.

NOTE: 'active_lines' counts lines that span the bar before the line's end

is knowable, a within-line lookahead inherited from the research design.

It is research-only and not computable live; the end-event columns are causal.

### `limen.features.price_range_position`

<!-- reference: api-limen-features-price-range-position source=limen/features/price_range_position.py region=signature:price_range_position role=signature -->
```python
def price_range_position(data: pl.DataFrame, period: int=24) -> pl.DataFrame:
    ...
```

Compute price position within rolling high-low range.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'high', 'low', 'close' columns |
| period (`int`) | Number of periods for rolling range calculation |

**Returns:** pl.DataFrame: The input data with a new column 'price_range_position'

### `limen.features.price_vs_band_regime`

<!-- reference: api-limen-features-price-vs-band-regime source=limen/features/price_vs_band_regime.py region=signature:price_vs_band_regime role=signature -->
```python
def price_vs_band_regime(df: pl.DataFrame, period: int=24, band: Literal['std', 'dev_std']='std', k: float=0.75) -> pl.DataFrame:
    ...
```

Compute regime by comparing 'close' to center ± k × band width over a rolling window.

| Argument | Contract |
| --- | --- |
| df (`pl.DataFrame`) | Klines dataset with 'close' column |
| period (`int`) | Rolling period for center and band width |
| band (`Literal['std', 'dev_std']`) | Band width type to use |
| k (`float`) | Band multiplier applied to the width |

**Returns:** pl.DataFrame: The input data with a new column 'regime_price_band'

### `limen.features.quantile_line_density`

<!-- reference: api-limen-features-quantile-line-density source=limen/features/quantile_line_density.py region=signature:quantile_line_density role=signature -->
```python
def quantile_line_density(data: pl.DataFrame, long_lines_q: list[dict[str, float]], short_lines_q: list[dict[str, float]], lookback_hours: int) -> pl.DataFrame:
    ...
```

Compute count of quantile-filtered line ends within lookback_hours window.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'datetime' and 'close' columns |
| long_lines_q (`list[dict]`) | Quantile-filtered long lines with 'end_idx' |
| short_lines_q (`list[dict]`) | Quantile-filtered short lines with 'end_idx' |
| lookback_hours (`int`) | Window size in hours for density calculation |

**Returns:** pl.DataFrame: The input data with a new column 'quantile_line_density_{lookback_hours}h'

### `limen.features.quantile_price_lines`

<!-- reference: api-limen-features-quantile-price-lines source=limen/features/quantile_price_lines.py region=signature:quantile_price_lines role=signature -->
```python
def quantile_price_lines(data: pl.DataFrame, max_duration_hours: int, min_height_pct: float, quantile_threshold: float=DEFAULT_QUANTILE_THRESHOLD, density_lookback_hours: int=DEFAULT_DENSITY_LOOKBACK_HOURS, momentum_lookback_hours: int=DEFAULT_MOMENTUM_LOOKBACK_HOURS, height_lookback_hours: int=DEFAULT_HEIGHT_LOOKBACK_HOURS, include_research_only: object=True) -> pl.DataFrame:
    ...
```

Compute quantile-line end features from lines completed through each row.

Select heights at or above each direction's expanding quantile, with ties.

End windows include both boundaries [t-window, t]; recency caps at density_lookback_hours.

Other empty windows return zero; active_quantile_count retains retrospective whole-frame spans.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame \| pl.LazyFrame`) | Klines with a close column |
| max_duration_hours (`int`) | Exclusive upper bound on line duration in bars |
| min_height_pct (`float`) | Minimum absolute height as a fraction of start price |
| quantile_threshold (`float`) | Per-direction height quantile |
| density_lookback_hours (`int`) | Density window and recency cap |
| momentum_lookback_hours (`int`) | Signed-height window |
| height_lookback_hours (`int`) | Mean-height and direction-bias window |
| include_research_only (`bool`) | Add the retrospective active_quantile_count |

**Returns:** pl.DataFrame \| pl.LazyFrame: Input type with five end columns and optional span count

**Raises:** ValueError: If a lookback is below 1 or include_research_only is not bool

### `limen.features.range_overlap`

<!-- reference: api-limen-features-range-overlap source=limen/features/range_overlap.py region=signature:range_overlap role=signature -->
```python
def range_overlap(data: pl.DataFrame, high_col: str='high', low_col: str='low') -> pl.DataFrame:
    ...
```

Compute how much the current bar overlaps the previous bar's range.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with high and low price columns |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |

**Returns:** pl.DataFrame: The input data with a new column 'range_overlap'

### `limen.features.range_pct`

<!-- reference: api-limen-features-range-pct source=limen/features/range_pct.py region=signature:range_pct role=signature -->
```python
def range_pct(data: pl.DataFrame) -> pl.DataFrame:
    ...
```

Compute range as percentage of close price (high-low)/close.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'high', 'low', and 'close' columns |

**Returns:** pl.DataFrame: The input data with a new column 'range_pct'

### `limen.features.range_per_dollar_volume`

<!-- reference: api-limen-features-range-per-dollar-volume source=limen/features/range_per_dollar_volume.py region=signature:range_per_dollar_volume role=signature -->
```python
def range_per_dollar_volume(data: pl.DataFrame, high_col: str='high', low_col: str='low', close_col: str='close', volume_col: str='volume') -> pl.DataFrame:
    ...
```

Compute normalized bar range per traded dollar.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with high, low, close, and volume columns |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |
| volume_col (`str`) | Column name for traded volume |

**Returns:** pl.DataFrame: The input data with a new column 'range_per_dollar_volume'

### `limen.features.realized_kurtosis`

<!-- reference: api-limen-features-realized-kurtosis source=limen/features/realized_kurtosis.py region=signature:realized_kurtosis role=signature -->
```python
def realized_kurtosis(data: pl.DataFrame, window: int=20, close_col: str='close') -> pl.DataFrame:
    ...
```

Compute rolling standardized kurtosis of close-to-close returns.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with a close price column |
| window (`int`) | Number of periods used for the rolling kurtosis estimate |
| close_col (`str`) | Column name used for close-to-close returns |

**Returns:** pl.DataFrame: The input data with a new column 'realized_kurtosis'

### `limen.features.realized_semivariance`

<!-- reference: api-limen-features-realized-semivariance source=limen/features/realized_semivariance.py region=signature:realized_semivariance role=signature -->
```python
def realized_semivariance(data: pl.DataFrame, window: int=20, close_col: str='close') -> pl.DataFrame:
    ...
```

Compute rolling upside and downside semivariance from close-to-close returns.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with a close price column |
| window (`int`) | Number of periods used for the rolling semivariance estimates |
| close_col (`str`) | Column name used for close-to-close returns |

**Returns:** pl.DataFrame: The input data with new columns 'upside_semivariance' and 'downside_semivariance'

### `limen.features.realized_skewness`

<!-- reference: api-limen-features-realized-skewness source=limen/features/realized_skewness.py region=signature:realized_skewness role=signature -->
```python
def realized_skewness(data: pl.DataFrame, window: int=20, close_col: str='close') -> pl.DataFrame:
    ...
```

Compute rolling standardized skewness of close-to-close returns.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with a close price column |
| window (`int`) | Number of periods used for the rolling skewness estimate |
| close_col (`str`) | Column name used for close-to-close returns |

**Returns:** pl.DataFrame: The input data with a new column 'realized_skewness'

### `limen.features.regime_multiplier`

<!-- reference: api-limen-features-regime-multiplier source=limen/features/regime_multiplier.py region=signature:regime_multiplier role=signature -->
```python
def regime_multiplier(data: pl.DataFrame, regime_low_volatility_multiplier: float=0.8, regime_normal_volatility_multiplier: float=1.0, regime_high_volatility_multiplier: float=1.2) -> pl.DataFrame:
    ...
```

Compute volatility regime-based multiplier for dynamic parameter adjustment.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'volatility_regime' column |
| regime_low_volatility_multiplier (`float`) | Multiplier for low volatility regime |
| regime_normal_volatility_multiplier (`float`) | Multiplier for normal volatility regime |
| regime_high_volatility_multiplier (`float`) | Multiplier for high volatility regime |

**Returns:** pl.DataFrame: The input data with a new column 'regime_multiplier'

### `limen.features.rejection_intensity`

<!-- reference: api-limen-features-rejection-intensity source=limen/features/rejection_intensity.py region=signature:rejection_intensity role=signature -->
```python
def rejection_intensity(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

Compute the wick fraction that rejects movement against the candle close direction.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with open, high, low, and close columns |
| open_col (`str`) | Column name for open prices |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |

**Returns:** pl.DataFrame: The input data with a new column 'rejection_intensity'

### `limen.features.relative_range_seasonality`

<!-- reference: api-limen-features-relative-range-seasonality source=limen/features/relative_range_seasonality.py region=signature:relative_range_seasonality role=signature -->
```python
def relative_range_seasonality(data: pl.DataFrame, datetime_col: str='datetime', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

Compute normalized bar range relative to the trailing mean for the same hour of the week.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with datetime, high, low, and close columns |
| datetime_col (`str`) | Column name for datetimes used to derive hour-of-week buckets |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |

**Returns:** pl.DataFrame: The input data with a new column 'relative_range_seasonality'

### `limen.features.relative_volatility_seasonality`

<!-- reference: api-limen-features-relative-volatility-seasonality source=limen/features/relative_volatility_seasonality.py region=signature:relative_volatility_seasonality role=signature -->
```python
def relative_volatility_seasonality(data: pl.DataFrame, datetime_col: str='datetime', close_col: str='close') -> pl.DataFrame:
    ...
```

Compute absolute return magnitude relative to the trailing mean for the same hour of the week.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with datetime and close columns |
| datetime_col (`str`) | Column name for datetimes used to derive hour-of-week buckets |
| close_col (`str`) | Column name used for close-to-close returns |

**Returns:** pl.DataFrame: The input data with a new column 'relative_volatility_seasonality'

### `limen.features.relative_volume_seasonality`

<!-- reference: api-limen-features-relative-volume-seasonality source=limen/features/relative_volume_seasonality.py region=signature:relative_volume_seasonality role=signature -->
```python
def relative_volume_seasonality(data: pl.DataFrame, datetime_col: str='datetime', volume_col: str='volume') -> pl.DataFrame:
    ...
```

Compute current volume relative to the trailing mean for the same hour of the week.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with datetime and volume columns |
| datetime_col (`str`) | Column name for datetimes used to derive hour-of-week buckets |
| volume_col (`str`) | Column name for traded volume |

**Returns:** pl.DataFrame: The input data with a new column 'relative_volume_seasonality'

### `limen.features.return_autocorrelation`

<!-- reference: api-limen-features-return-autocorrelation source=limen/features/return_autocorrelation.py region=signature:return_autocorrelation role=signature -->
```python
def return_autocorrelation(data: pl.DataFrame, window: int=20, close_col: str='close', output_col: str='return_autocorrelation') -> pl.DataFrame:
    ...
```

Compute rolling autocorrelation between returns and one-bar lagged returns.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with a close price column |
| window (`int`) | Number of periods in the rolling correlation |
| close_col (`str`) | Column name for close prices |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the return-autocorrelation column appended

### `limen.features.return_per_dollar_volume`

<!-- reference: api-limen-features-return-per-dollar-volume source=limen/features/return_per_dollar_volume.py region=signature:return_per_dollar_volume role=signature -->
```python
def return_per_dollar_volume(data: pl.DataFrame, price_col: str='close', volume_col: str='volume') -> pl.DataFrame:
    ...
```

Compute signed return per traded dollar as a simple impact-efficiency proxy.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with price and volume columns |
| price_col (`str`) | Column name used for close-to-close returns |
| volume_col (`str`) | Column name for traded volume |

**Returns:** pl.DataFrame: The input data with a new column 'return_per_dollar_volume'

### `limen.features.return_volatility_correlation`

<!-- reference: api-limen-features-return-volatility-correlation source=limen/features/return_volatility_correlation.py region=signature:return_volatility_correlation role=signature -->
```python
def return_volatility_correlation(data: pl.DataFrame, window: int=20, close_col: str='close', high_col: str='high', low_col: str='low', output_col: str='return_volatility_correlation') -> pl.DataFrame:
    ...
```

Compute rolling correlation between returns and Parkinson variance.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with close, high, and low columns |
| window (`int`) | Number of periods in the rolling correlation |
| close_col (`str`) | Column name for close prices |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the return-volatility correlation appended

### `limen.features.returns_lags`

<!-- reference: api-limen-features-returns-lags source=limen/features/returns_lags.py region=signature:returns_lags role=signature -->
```python
def returns_lags(data: pl.DataFrame, max_lag: int=24, returns_col: str='returns') -> pl.DataFrame:
    ...
```

Compute multiple lagged returns features for time series analysis.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with returns column |
| max_lag (`int`) | Maximum number of lag periods to compute |
| returns_col (`str`) | Name of the returns column |

**Returns:** pl.DataFrame: The input data with new columns 'returns_lag_1', 'returns_lag_2', etc.

### `limen.features.rogers_satchell_volatility`

<!-- reference: api-limen-features-rogers-satchell-volatility source=limen/features/rogers_satchell_volatility.py region=signature:rogers_satchell_volatility role=signature -->
```python
def rogers_satchell_volatility(data: pl.DataFrame, window: int=20, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

Compute Rogers-Satchell range-based volatility over a rolling window.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with open, high, low, and close price columns |
| window (`int`) | Number of periods used for the rolling estimator |
| open_col (`str`) | Column name for open prices |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |

**Returns:** pl.DataFrame: The input data with a new column 'rogers_satchell_volatility'

### `limen.features.rolling_zscore`

<!-- reference: api-limen-features-rolling-zscore source=limen/features/rolling_zscore.py region=signature:rolling_zscore role=signature -->
```python
def rolling_zscore(data: pl.DataFrame, column: str, window: int, *, transform: ZscoreTransform='identity', output_col: str | None=None) -> pl.DataFrame:
    ...
```

Compute a rolling z-score for a transformed input column.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset containing the source column |
| column (`str`) | Source column name |
| window (`int`) | Number of periods for the rolling mean and standard deviation |
| transform (`ZscoreTransform`) | Optional source transform before standardization |
| output_col (`str \| None`) | Output column name; defaults to `{column}_zscore_{window}` |

**Returns:** pl.DataFrame: The input data with the rolling z-score column appended

### `limen.features.sma_crossover`

<!-- reference: api-limen-features-sma-crossover source=limen/features/sma_crossover.py region=signature:sma_crossover role=signature -->
```python
def sma_crossover(df: pl.DataFrame, short_window: int=10, long_window: int=30, crossover_bull: int=2, crossover_bear: int=-2) -> pl.DataFrame:
    ...
```

Compute Simple Moving Average (SMA) crossover signals.

| Argument | Contract |
| --- | --- |
| df (`pl.DataFrame`) | Klines dataset with 'close' column |
| short_window (`int`) | Number of periods for short-term SMA |
| long_window (`int`) | Number of periods for long-term SMA |
| crossover_bull (`int`) | Value indicating bullish crossover |
| crossover_bear (`int`) | Value indicating bearish crossover |

**Returns:** pl.DataFrame: The input data with new columns 'crossover', and 'signal'

### `limen.features.sma_ratios`

<!-- reference: api-limen-features-sma-ratios source=limen/features/sma_ratios.py region=signature:sma_ratios role=signature -->
```python
def sma_ratios(data: pl.DataFrame, periods: list[int] | None=None, price_col: str='close') -> pl.DataFrame:
    ...
```

Compute price to SMA ratios for multiple time periods.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with price column |
| periods (`list`) | List of SMA periods to calculate ratios for |
| price_col (`str`) | Name of the price column (default: 'close') |

**Returns:** pl.DataFrame: The input data with new columns 'sma_{period}_ratio'

### `limen.features.spread`

<!-- reference: api-limen-features-spread source=limen/features/spread.py region=signature:spread role=signature -->
```python
def spread(df: pl.DataFrame) -> pl.DataFrame:
    ...
```

Compute price spread as percentage of close price.

| Argument | Contract |
| --- | --- |
| df (`pl.DataFrame`) | Klines dataset with 'high', 'low', 'close' columns |

**Returns:** pl.DataFrame: The input data with a new column 'spread'

### `limen.features.spread_percent`

<!-- reference: api-limen-features-spread-percent source=limen/features/spread_percent.py region=signature:spread_percent role=signature -->
```python
def spread_percent(data: pl.DataFrame) -> pl.DataFrame:
    ...
```

Compute high-low spread as percentage of close price.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'high', 'low', 'close' columns |

**Returns:** pl.DataFrame: The input data with a new column 'spread_percent'

### `limen.features.stochastic_k_abs`

<!-- reference: api-limen-features-stochastic-k-abs source=limen/features/stochastic_k_abs.py region=signature:stochastic_k_abs role=signature -->
```python
def stochastic_k_abs(data: pl.DataFrame, window: int=14, high_col: str='high', low_col: str='low', close_col: str='close', output_col: str='stochastic_k_abs') -> pl.DataFrame:
    ...
```

Compute absolute distance of stochastic percent K from the center line.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with high, low, and close columns |
| window (`int`) | Number of periods in the high-low channel |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the stochastic distance column appended

### `limen.features.tail_event_intensity`

<!-- reference: api-limen-features-tail-event-intensity source=limen/features/tail_event_intensity.py region=signature:tail_event_intensity role=signature -->
```python
def tail_event_intensity(data: pl.DataFrame, window: int=20, z_threshold: float=2.0, close_col: str='close') -> pl.DataFrame:
    ...
```

Compute the trailing share of return shocks that exceed a volatility-scaled threshold.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with a close price column |
| window (`int`) | Number of periods used for both the volatility baseline and event share |
| z_threshold (`float`) | Multiple of trailing return volatility required to count as a tail event |
| close_col (`str`) | Column name used for close-to-close returns |

**Returns:** pl.DataFrame: The input data with a new column 'tail_event_intensity'

### `limen.features.taker_imbalance_ratio`

<!-- reference: api-limen-features-taker-imbalance-ratio source=limen/features/taker_imbalance_ratio.py region=signature:taker_imbalance_ratio role=signature -->
```python
def taker_imbalance_ratio(data: pl.DataFrame, window: int=20, volume_col: str='volume', maker_ratio_col: str='maker_ratio', output_col: str='taker_imbalance_ratio') -> pl.DataFrame:
    ...
```

Compute rolling absolute taker imbalance as a share of volume.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with volume and maker-ratio columns |
| window (`int`) | Number of periods in the rolling mean |
| volume_col (`str`) | Column name for total traded volume |
| maker_ratio_col (`str`) | Column name for maker-side volume ratio |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the taker-imbalance ratio column appended

### `limen.features.time_to_funding`

<!-- reference: api-limen-features-time-to-funding source=limen/features/time_to_funding.py region=signature:time_to_funding role=signature -->
```python
def time_to_funding(data: pl.DataFrame, interval_hours: int=8, offset_hour: int=0, datetime_col: str='datetime', output_col: str='hours_to_funding') -> pl.DataFrame:
    ...
```

Compute the number of hours until the next funding settlement.

Funding settles every `interval_hours` from `offset_hour` UTC (the default

cadence is 0, 8, and 16 hours). The output is the continuous distance in hours

from each timestamp to the next settlement, and is zero at a settlement bar.

The cadence is anchored to UTC midnight, so `interval_hours` must divide 24.

Timezone-aware timestamps are converted to UTC; naive timestamps are assumed

to already be in UTC.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Dataset with a datetime column |
| interval_hours (`int`) | Spacing between funding settlements in hours; must divide 24 |
| offset_hour (`int`) | Hour of the first daily settlement in UTC |
| datetime_col (`str`) | Column name for timestamps |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with a new 'hours_to_funding' column

### `limen.features.trade_density`

<!-- reference: api-limen-features-trade-density source=limen/features/trade_density.py region=signature:trade_density role=signature -->
```python
def trade_density(data: pl.DataFrame, window: int=20, trades_col: str='no_of_trades', volume_col: str='volume', output_col: str='trade_density') -> pl.DataFrame:
    ...
```

Compute rolling mean trade count per unit of volume.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with trade-count and volume columns |
| window (`int`) | Number of periods in the rolling mean |
| trades_col (`str`) | Column name for number of trades |
| volume_col (`str`) | Column name for total traded volume |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the trade-density column appended

### `limen.features.trade_imbalance`

<!-- reference: api-limen-features-trade-imbalance source=limen/features/trade_imbalance.py region=signature:trade_imbalance role=signature -->
```python
def trade_imbalance(data: pl.DataFrame, window: int=20, maker_volume_col: str='maker_volume', volume_col: str='volume', output_col: str='trade_imbalance') -> pl.DataFrame:
    ...
```

Compute rolling maker volume divided by rolling total volume.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with maker volume and total volume columns |
| window (`int`) | Number of periods used for rolling sums |
| maker_volume_col (`str`) | Column name for maker-side traded volume |
| volume_col (`str`) | Column name for total traded volume |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the rolling trade-imbalance column appended

### `limen.features.trade_size_ratio`

<!-- reference: api-limen-features-trade-size-ratio source=limen/features/trade_size_ratio.py region=signature:trade_size_ratio role=signature -->
```python
def trade_size_ratio(data: pl.DataFrame, short_window: int=5, long_window: int=20, volume_col: str='volume', trades_col: str='no_of_trades', output_col: str='trade_size_ratio') -> pl.DataFrame:
    ...
```

Compare short and long rolling average trade sizes.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with volume and trade-count columns |
| short_window (`int`) | Number of periods in the short average |
| long_window (`int`) | Number of periods in the long average |
| volume_col (`str`) | Column name for total traded volume |
| trades_col (`str`) | Column name for number of trades |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the trade-size ratio column appended

### `limen.features.trend_coherence`

<!-- reference: api-limen-features-trend-coherence source=limen/features/trend_coherence.py region=signature:trend_coherence role=signature -->
```python
def trend_coherence(data: pl.DataFrame, short_window: int=12, medium_window: int=48, long_window: int=168, close_col: str='close') -> pl.DataFrame:
    ...
```

Compute directional agreement across short, medium, and long return horizons.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with a close price column |
| short_window (`int`) | Short return horizon |
| medium_window (`int`) | Medium return horizon |
| long_window (`int`) | Long return horizon |
| close_col (`str`) | Column name used for return calculations |

**Returns:** pl.DataFrame: The input data with a new column 'trend_coherence'

### `limen.features.trend_strength`

<!-- reference: api-limen-features-trend-strength source=limen/features/trend_strength.py region=signature:trend_strength role=signature -->
```python
def trend_strength(data: pl.DataFrame, fast_period: int=20, slow_period: int=50) -> pl.DataFrame:
    ...
```

Compute trend strength based on moving average divergence.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'close' column |
| fast_period (`int`) | Number of periods for fast SMA calculation |
| slow_period (`int`) | Number of periods for slow SMA calculation |

**Returns:** pl.DataFrame: The input data with a new column 'trend_strength'

### `limen.features.volatility_1h`

<!-- reference: api-limen-features-volatility-1h source=limen/features/volatility_1h.py region=signature:volatility_1h role=signature -->
```python
def volatility_1h(df: pl.DataFrame, volatility_column: str='returns_volatility_12') -> pl.DataFrame:
    ...
```

Compute volatility_1h alias from existing volatility column.

| Argument | Contract |
| --- | --- |
| df (`pl.DataFrame`) | Klines dataset with volatility column |
| volatility_column (`str`) | Name of the source volatility column |

**Returns:** pl.DataFrame: The input data with a new column 'volatility_1h'

### `limen.features.volatility_measure`

<!-- reference: api-limen-features-volatility-measure source=limen/features/volatility_measure.py region=signature:volatility_measure role=signature -->
```python
def volatility_measure(data: pl.DataFrame) -> pl.DataFrame:
    ...
```

Compute combined volatility measure from rolling volatility and ATR percentage.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'rolling_volatility', 'atr_percent_sma' columns |

**Returns:** pl.DataFrame: The input data with a new column 'volatility_measure'

### `limen.features.volatility_of_volatility`

<!-- reference: api-limen-features-volatility-of-volatility source=limen/features/volatility_of_volatility.py region=signature:volatility_of_volatility role=signature -->
```python
def volatility_of_volatility(data: pl.DataFrame, volatility_window: int=12, window: int=48, close_col: str='close') -> pl.DataFrame:
    ...
```

Compute how unstable recent return volatility is over a second rolling horizon.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with a close price column |
| volatility_window (`int`) | Number of periods used for the first-stage return volatility |
| window (`int`) | Number of periods used for the second-stage volatility-of-volatility estimate |
| close_col (`str`) | Column name used for close-to-close returns |

**Returns:** pl.DataFrame: The input data with a new column 'volatility_of_volatility'

### `limen.features.volatility_ratio`

<!-- reference: api-limen-features-volatility-ratio source=limen/features/volatility_ratio.py region=signature:volatility_ratio role=signature -->
```python
def volatility_ratio(data: pl.DataFrame, short_window: int=5, long_window: int=20, high_col: str='high', low_col: str='low', output_col: str='volatility_ratio') -> pl.DataFrame:
    ...
```

Compare short and long rolling means of Parkinson variance.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with high and low price columns |
| short_window (`int`) | Number of periods in the short variance mean |
| long_window (`int`) | Number of periods in the long variance mean |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the volatility-ratio column appended

### `limen.features.volatility_spike`

<!-- reference: api-limen-features-volatility-spike source=limen/features/volatility_spike.py region=signature:volatility_spike role=signature -->
```python
def volatility_spike(data: pl.DataFrame, window: int=20, high_col: str='high', low_col: str='low', output_col: str='volatility_spike') -> pl.DataFrame:
    ...
```

Compute current Parkinson variance relative to its fixed-lag value.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with high and low price columns |
| window (`int`) | Lag used as the denominator |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the volatility-spike column appended

### `limen.features.volatility_term_structure`

<!-- reference: api-limen-features-volatility-term-structure source=limen/features/volatility_term_structure.py region=signature:volatility_term_structure role=signature -->
```python
def volatility_term_structure(data: pl.DataFrame, short_window: int=12, medium_window: int=48, long_window: int=168, close_col: str='close') -> pl.DataFrame:
    ...
```

Compute a front-versus-back volatility term-structure ratio across three horizons.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with a close price column |
| short_window (`int`) | Short volatility horizon |
| medium_window (`int`) | Medium volatility horizon |
| long_window (`int`) | Long volatility horizon |
| close_col (`str`) | Column name used for close-to-close returns |

**Returns:** pl.DataFrame: The input data with a new column 'volatility_term_structure'

### `limen.features.volatility_weight`

<!-- reference: api-limen-features-volatility-weight source=limen/features/volatility_weight.py region=signature:volatility_weight role=signature -->
```python
def volatility_weight(data: pl.DataFrame, period: int=20, volatility_scaling_factor: float=100, volatility_weight_min: float=0.3, volatility_weight_max: float=1.0) -> pl.DataFrame:
    ...
```

Compute volatility-based weighting factor with inverse scaling.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'close' column |
| period (`int`) | Period for volatility calculation |
| volatility_scaling_factor (`float`) | Scaling factor for volatility normalization |
| volatility_weight_min (`float`) | Minimum volatility weight value |
| volatility_weight_max (`float`) | Maximum volatility weight value |

**Returns:** pl.DataFrame: The input data with new columns 'volatility', 'volatility_weight'

### `limen.features.volume_ratio`

<!-- reference: api-limen-features-volume-ratio source=limen/features/volume_ratio.py region=signature:volume_ratio role=signature -->
```python
def volume_ratio(data: pl.DataFrame, period: int=20) -> pl.DataFrame:
    ...
```

Compute volume ratio relative to simple moving average baseline.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'volume' column |
| period (`int`) | Number of periods for SMA calculation |

**Returns:** pl.DataFrame: The input data with a new column 'volume_ratio'

### `limen.features.volume_regime`

<!-- reference: api-limen-features-volume-regime source=limen/features/volume_regime.py region=signature:volume_regime role=signature -->
```python
def volume_regime(data: pl.DataFrame, lookback: int=48) -> pl.DataFrame:
    ...
```

Compute volume regime (current vs average volume).

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'volume' column |
| lookback (`int`) | Number of periods for volume average calculation |

**Returns:** pl.DataFrame: The input data with a new column 'volume_regime'

### `limen.features.volume_spike`

<!-- reference: api-limen-features-volume-spike source=limen/features/volume_spike.py region=signature:volume_spike role=signature -->
```python
def volume_spike(data: pl.DataFrame, period: int=20, use_zscore: bool=False) -> pl.DataFrame:
    ...
```

Compute volume spike relative to rolling statistics baseline.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'volume' column |
| period (`int`) | Number of periods for rolling statistics calculation |
| use_zscore (`bool`) | If True, use z-score (incorporating std dev), otherwise use ratio to mean |

**Returns:** pl.DataFrame: The input data with a new column 'volume_spike'

### `limen.features.volume_to_range`

<!-- reference: api-limen-features-volume-to-range source=limen/features/volume_to_range.py region=signature:volume_to_range role=signature -->
```python
def volume_to_range(data: pl.DataFrame, window: int=20, volume_col: str='volume', high_col: str='high', low_col: str='low', output_col: str='volume_to_range') -> pl.DataFrame:
    ...
```

Compute rolling mean volume per unit of high-low range.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with volume, high, and low columns |
| window (`int`) | Number of periods in the rolling mean |
| volume_col (`str`) | Column name for traded volume |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the volume-to-range column appended

### `limen.features.volume_trend`

<!-- reference: api-limen-features-volume-trend source=limen/features/volume_trend.py region=signature:volume_trend role=signature -->
```python
def volume_trend(data: pl.DataFrame, short_period: int=12, long_period: int=48) -> pl.DataFrame:
    ...
```

Compute volume trend by comparing short-term to long-term volume averages.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with 'volume' column |
| short_period (`int`) | Number of periods for short-term SMA calculation |
| long_period (`int`) | Number of periods for long-term SMA calculation |

**Returns:** pl.DataFrame: The input data with a new column 'volume_trend'

### `limen.features.volume_volatility_correlation`

<!-- reference: api-limen-features-volume-volatility-correlation source=limen/features/volume_volatility_correlation.py region=signature:volume_volatility_correlation role=signature -->
```python
def volume_volatility_correlation(data: pl.DataFrame, window: int=20, volume_col: str='volume', high_col: str='high', low_col: str='low', output_col: str='volume_volatility_correlation') -> pl.DataFrame:
    ...
```

Compute rolling correlation between volume and Parkinson variance.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with volume, high, and low columns |
| window (`int`) | Number of periods in the rolling correlation |
| volume_col (`str`) | Column name for traded volume |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the volume-volatility correlation appended

### `limen.features.vpin`

<!-- reference: api-limen-features-vpin source=limen/features/vpin.py region=signature:vpin role=signature -->
```python
def vpin(data: pl.DataFrame, window: int=50, classification_window: int=20, close_col: str='close', volume_col: str='volume') -> pl.DataFrame:
    ...
```

Compute Volume-synchronized Probability of Informed Trading (VPIN).

Bars are split into buy and sell volume by bulk-volume classification, then

VPIN is the absolute buy-minus-sell imbalance summed over `window` buckets

divided by the summed total volume (Easley, Lopez de Prado, and O'Hara). High

values flag one-sided, toxic-looking flow. For canonical equal-volume buckets,

supply volume bars; on time bars this is the standard bar-level approximation.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with close and volume columns |
| window (`int`) | Number of buckets over which the imbalance is accumulated |
| classification_window (`int`) | Rolling window for the bulk-volume classification standardization |
| close_col (`str`) | Column name used for close-to-close log returns |
| volume_col (`str`) | Column name for total traded volume |

**Returns:** pl.DataFrame: The input data with classified buy/sell volume columns and a new 'vpin' column

### `limen.features.vwap`

<!-- reference: api-limen-features-vwap source=limen/features/vwap.py region=signature:vwap role=signature -->
```python
def vwap(data: pl.DataFrame, price_col: str='close', volume_col: str='volume') -> pl.DataFrame:
    ...
```

Compute Volume Weighted Average Price (VWAP) for each kline over its trading day.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with price and volume columns |
| price_col (`str`) | Name of the price column |
| volume_col (`str`) | Name of the volume column |

**Returns:** pl.DataFrame: The input data with a new column 'vwap'

### `limen.features.wick_imbalance`

<!-- reference: api-limen-features-wick-imbalance source=limen/features/wick_imbalance.py region=signature:wick_imbalance role=signature -->
```python
def wick_imbalance(data: pl.DataFrame, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

Compute upper-versus-lower wick imbalance as a fraction of total bar range.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with open, high, low, and close columns |
| open_col (`str`) | Column name for open prices |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |

**Returns:** pl.DataFrame: The input data with a new column 'wick_imbalance'

### `limen.features.wick_proportion`

<!-- reference: api-limen-features-wick-proportion source=limen/features/wick_proportion.py region=signature:wick_proportion role=signature -->
```python
def wick_proportion(data: pl.DataFrame, window: int=20, high_col: str='high', low_col: str='low', open_col: str='open', close_col: str='close', output_col: str='wick_proportion') -> pl.DataFrame:
    ...
```

Compute rolling mean wick share of the full candle range.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with open, high, low, and close columns |
| window (`int`) | Number of periods in the rolling mean |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| open_col (`str`) | Column name for open prices |
| close_col (`str`) | Column name for close prices |
| output_col (`str`) | Output column name |

**Returns:** pl.DataFrame: The input data with the wick-proportion column appended

### `limen.features.window_return_regime`

<!-- reference: api-limen-features-window-return-regime source=limen/features/window_return_regime.py region=signature:window_return_regime role=signature -->
```python
def window_return_regime(df: pl.DataFrame, period: int=24, r_hi: float=0.0, r_lo: float=0.0) -> pl.DataFrame:
    ...
```

Compute regime using windowed return close/close.shift(period) - 1.

| Argument | Contract |
| --- | --- |
| df (`pl.DataFrame`) | Klines dataset with 'close' column |
| period (`int`) | Window length for return calculation |
| r_hi (`float`) | Upper threshold for Up regime |
| r_lo (`float`) | Lower threshold for Down regime |

**Returns:** pl.DataFrame: The input data with a new column 'regime_window_return'

### `limen.features.yang_zhang_volatility`

<!-- reference: api-limen-features-yang-zhang-volatility source=limen/features/yang_zhang_volatility.py region=signature:yang_zhang_volatility role=signature -->
```python
def yang_zhang_volatility(data: pl.DataFrame, window: int=20, open_col: str='open', high_col: str='high', low_col: str='low', close_col: str='close') -> pl.DataFrame:
    ...
```

Compute Yang-Zhang volatility over a rolling window.

| Argument | Contract |
| --- | --- |
| data (`pl.DataFrame`) | Klines dataset with open, high, low, and close price columns |
| window (`int`) | Number of periods used for the rolling estimator; must be greater than 1 |
| open_col (`str`) | Column name for open prices |
| high_col (`str`) | Column name for high prices |
| low_col (`str`) | Column name for low prices |
| close_col (`str`) | Column name for close prices |

**Returns:** pl.DataFrame: The input data with a new column 'yang_zhang_volatility'
