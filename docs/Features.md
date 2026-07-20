# Features

Features sit one layer above indicators. They combine, re-express, lag, or contextualize price and volume information into model-ready signals or regime flags.

This page covers feature-helper selection, input requirements, and added columns. For lower-level signal primitives, see [Indicators](Indicators.md).

## Conventions

- Feature helpers append columns and return the original frame with the extra outputs attached.
- Helpers either operate on plain kline data or require trade-derived columns such as `maker_ratio`, `no_of_trades`, `price`, or `quantity`.
- Not every helper is intended as a final predictor. Some are utilities used to build targets or wider feature families.
- Regime helpers output compact categorical-style columns such as `regime_ma_slope` or `regime_price_band`.
- `find_min_d` returns a scalar value instead of a frame; it searches for a stationarity-preserving fractional-differentiation order.

## Quick example

```python
from limen.data import HistoricalData
from limen.experiment import MLManifest
from limen.features import kline_imbalance, vwap
from limen.indicators import atr, roc
from limen.sfd.reference_architecture import logreg_binary


def manifest():
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
        .add_feature(vwap)
        .add_feature(kline_imbalance, window='imbalance_window')
        .with_reference_architecture(logreg_binary)
    )
```

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

The two grouped transforms below are the YAML-facing surface. Each detects lines internally from scalar params on the frame it receives — per split under the manifest pipeline, so detection cannot observe other splits — and adds its full column family in one detection pass. Line detection lives in `limen.utils.find_price_lines` / `limen.utils.filter_lines_by_quantile`.

| Function | Adds by default | Notes |
|---|---|---|
| `price_lines` | `active_lines`, `hours_since_big_move`, `line_momentum_<m>h`, `trending_score`, `reversal_potential` | All-line family: span count, end recency (capped at `big_move_lookback_hours`), and long-minus-short end counts over the trailing `[t-m, t)` window with their balance (`trending_score`, in `[-1, 1]`) and min/max ratio (`reversal_potential`, in `[0, 1]`). |
| `quantile_price_lines` | `hours_since_quantile_line`, `active_quantile_count`, `quantile_line_density_<d>h`, `quantile_momentum_<m>h`, `avg_quantile_height_<h>h`, `quantile_direction_bias` | Quantile-line family: end recency and span count, end density over `density_lookback_hours`, signed height sum over ends in `[t-m, t]`, and mean height plus height-weighted direction (in `[-1, 1]`) over ends in `[t-h, t]`. |

`active_lines` and `active_quantile_count` count lines that span the current bar before the line's end — the event that defines it — is knowable. That within-line lookahead is inherited from the tradeline research design: treat both columns as research-only, not live-computable. The grouped transforms expose `include_research_only`; set it to `false` to omit the active-span columns. Bundled live-safe templates set `include_research_only: false`. The end-event columns are causal.

The per-column building blocks below take pre-computed line structures (`list[dict]` with `start_idx`/`end_idx`) and remain available for programmatic composition; the grouped transforms above compose them.

| Function | Adds by default | Notes |
|---|---|---|
| `active_lines` | `active_lines` | Count of long and short lines active at each bar. |
| `active_quantile_count` | `active_quantile_count` | Active-line count restricted to quantile-filtered lines. |
| `quantile_line_density` | `quantile_line_density_<lookback>h` | Count of quantile-line endings within a trailing `lookback_hours` window. |
| `hours_since_big_move` | `hours_since_big_move` | Bars since the most recent line end, capped at `lookback_hours`. |
| `hours_since_quantile_line` | `hours_since_quantile_line` | Bars since the most recent quantile-line end, capped at `lookback_hours`. |

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
| `market_regime` | `sma_20`, `sma_50`, `trend_strength`, `volatility_ratio`, `volume_sma`, `volume_regime`, `market_favorable` | Trend-strength and volume-regime favorability score. |
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
