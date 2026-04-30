# Features

Features sit one layer above indicators. They usually combine, re-express, lag, or contextualize price and volume information into model-ready signals or regime flags.

Use this page when you need to choose between feature helpers, understand their input requirements, or see what each helper adds to a frame. For lower-level signal primitives, see [Indicators](Indicators.md).

## Conventions

- Feature helpers usually append columns and return the original frame with the extra outputs attached.
- Some helpers operate on plain kline data only. Others require trade-derived columns such as `maker_ratio`, `no_of_trades`, `price`, or `quantity`.
- Not every helper is intended as a final predictor. Some are utilities used to build targets or wider feature families.
- Several regime helpers output compact categorical-style columns such as `regime_ma_slope` or `regime_price_band`.
- A few helpers return scalar values instead of frames. The main example is `find_min_d`, which searches for a stationarity-preserving fractional-differentiation order.

## Quick Example

```python
from limen.data import HistoricalData
from limen.experiment import Manifest
from limen.features import kline_imbalance, vwap
from limen.indicators import atr, roc
from limen.sfd.reference_architecture import logreg_binary


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
        .add_feature(vwap)
        .add_feature(kline_imbalance, window='imbalance_window')
        .with_reference_architecture(logreg_binary)
    )
```

## Kline-Derived Position And Volatility Features

These helpers work directly on bar data and are usually the easiest feature layer to add to a first manifest.

| Function | Adds by default | Notes |
|---|---|---|
| `atr_percent_sma` | `atr_percent_sma` | ATR scaled by close price using SMA smoothing. |
| `atr_sma` | `atr_sma` | SMA-smoothed ATR variant. |
| `close_position` | `close_position` | Close location inside the current bar's high-low range. |
| `distance_from_high` | `distance_from_high` | Distance from a rolling high. |
| `distance_from_low` | `distance_from_low` | Distance from a rolling low. |
| `gap_high` | `gap_high` | Current high relative to the previous close. |
| `price_range_position` | `price_range_position` | Rolling range position over a wider window. |
| `range_pct` | `range_pct` | Current bar range as a percentage. |
| `trend_strength` | `trend_strength` | Fast-versus-slow trend strength summary. |
| `volume_regime` | `volume_regime` | Volume context over a lookback window. |
| `vwap` | `vwap` | Requires a datetime-like `datetime` column because VWAP resets by trading day. |

## Calendar And Cyclical Time Features

These helpers derive time-of-bar context from `datetime` without depending on earlier indicator columns.

| Function | Adds by default | Notes |
|---|---|---|
| `calendar_time_features` | `hour`, `minute`, `weekday`, `day_of_month`, `day_of_year`, `week_of_year`, `month`, `quarter`, `is_weekend` | Adds discrete calendar fields for downstream splits, filters, and rules. `weekday` uses ISO numbering (`Monday=1` to `Sunday=7`), and `week_of_year` uses ISO week numbering. |
| `cyclical_time_features` | `hour_sin`, `hour_cos`, `minute_sin`, `minute_cos`, `weekday_sin`, `weekday_cos`, `day_of_month_sin`, `day_of_month_cos`, `day_of_year_sin`, `day_of_year_cos`, `week_of_year_sin`, `week_of_year_cos`, `month_sin`, `month_cos`, `quarter_sin`, `quarter_cos` | Encodes cyclical calendar fields without introducing artificial ordinal jumps. Uses the same ISO conventions as `calendar_time_features`; weekday cycles are phase-aligned with `weekday - 1` before applying sine/cosine. |

## Range-Based Volatility Features

These helpers estimate volatility directly from OHLC structure instead of relying only on close-to-close returns.

| Function | Adds by default | Notes |
|---|---|---|
| `parkinson_volatility` | `parkinson_volatility` | High-low range estimator that ignores close-to-close drift. |
| `garman_klass_volatility` | `garman_klass_volatility` | OHLC estimator using open, high, low, and close information. |
| `rogers_satchell_volatility` | `rogers_satchell_volatility` | Drift-robust OHLC volatility estimator. |
| `yang_zhang_volatility` | `yang_zhang_volatility` | Combines overnight, open-close, and range information into a higher-fidelity volatility estimate. Requires `window > 1`. |

## Liquidity And Impact Features

These helpers translate ordinary OHLCV bars into simple liquidity, impact, and slippage proxies.

| Function | Adds by default | Notes |
|---|---|---|
| `dollar_volume` | `dollar_volume` | Price-times-volume activity proxy using close and volume. |
| `amihud_illiquidity` | `amihud_illiquidity` | Absolute return per dollar of volume, a compact price-impact proxy. |
| `return_per_dollar_volume` | `return_per_dollar_volume` | Signed return per dollar of volume, useful when direction matters as much as impact. |
| `range_per_dollar_volume` | `range_per_dollar_volume` | Bar range scaled by dollar volume. |
| `illiquidity_shock` | `illiquidity_shock` | Current Amihud-style illiquidity relative to its own trailing mean. |

## Realized Risk And Tail Features

These helpers describe the quality of recent movement, not just its level.

| Function | Adds by default | Notes |
|---|---|---|
| `realized_semivariance` | `upside_semivariance`, `downside_semivariance` | Splits rolling squared returns into upside and downside components. |
| `realized_skewness` | `realized_skewness` | Rolling skewness of close-to-close returns. |
| `realized_kurtosis` | `realized_kurtosis` | Rolling kurtosis of close-to-close returns. |
| `jump_variation_proxy` | `jump_variation_proxy` | Positive gap between realized variance and bipower variation proxy. |
| `tail_event_intensity` | `tail_event_intensity` | Share of recent bars whose absolute return exceeds a configurable threshold. |
| `volatility_of_volatility` | `volatility_of_volatility` | Rolling variability of rolling close-to-close return volatility. |

## Seasonality-Normalized Features

These helpers compare current bar behavior to the trailing mean for the same hour of the week.

| Function | Adds by default | Notes |
|---|---|---|
| `relative_volume_seasonality` | `relative_volume_seasonality` | Current volume relative to the trailing baseline for the same hour-of-week bucket. |
| `relative_range_seasonality` | `relative_range_seasonality` | Current range percentage relative to the trailing hour-of-week baseline. |
| `relative_volatility_seasonality` | `relative_volatility_seasonality` | Current absolute return magnitude relative to the trailing hour-of-week baseline. |

## Candle Structure And Auction Features

These helpers focus on how a bar moved internally, not just where it finished.

| Function | Adds by default | Notes |
|---|---|---|
| `body_to_range` | `body_to_range` | Absolute candle body size divided by the full bar range. |
| `wick_imbalance` | `wick_imbalance` | Upper-wick minus lower-wick imbalance as a share of full range. |
| `range_overlap` | `range_overlap` | Overlap share between the current bar range and the previous bar range. |
| `rejection_intensity` | `rejection_intensity` | Wick-heavy rejection proxy based on total wick share and directional close location. |
| `absorption_intensity` | `absorption_intensity` | High-volume, small-body absorption proxy using a trailing shifted volume baseline. |

## Cross-Timescale Context Features

These helpers summarize whether multiple horizons agree or disagree about market state.

| Function | Adds by default | Notes |
|---|---|---|
| `trend_coherence` | `trend_coherence` | Average sign agreement across short, medium, and long return horizons. |
| `volatility_term_structure` | `volatility_term_structure` | Average ratio between short, medium, and long rolling volatility estimates. |

## Breakout And Regime Features

These helpers are useful when you want state or structure, not just a continuous numeric series.

| Function | Adds by default | Notes |
|---|---|---|
| `breakout_features` | many lagged breakout columns plus `long_roll_mean`, `long_roll_std`, `short_roll_mean`, `short_roll_std`, `roc_long_12_1`, `roc_short_12_1` | Designed to enrich pre-existing breakout flags. |
| `breakout_percentile_regime` | `price_range_position`, `regime_breakout_pct` | Uses percentile thresholds over price-range position. |
| `ema_breakout` | `breakout_ema` | Requires `target_col`; flags price displacement from EMA beyond `breakout_delta`. |
| `hh_hl_structure_regime` | `regime_hh_hl` | Captures higher-high and higher-low style structure. |
| `ichimoku_cloud` | `tenkan`, `kijun`, `senkou_a`, `senkou_b`, `chikou` | Full Ichimoku feature set. |
| `ma_slope_regime` | `regime_ma_slope` | Regime label based on moving-average slope. |
| `price_vs_band_regime` | `regime_price_band` | Uses price distance relative to a band definition. |
| `sma_crossover` | `crossover`, `signal` | Compact crossover-state helper. |
| `window_return_regime` | `ret_24`, `regime_window_return` | Return plus regime thresholding over a window. |

## Lag Helpers And Threshold Utilities

These helpers are mainly used to expand existing columns or define cutoffs for target construction.

| Function | Adds or returns | Notes |
|---|---|---|
| `lag_column` | one lagged column such as `close_lag_2` | Requires `col` and `lag`. |
| `lag_columns` | one lag per listed column | Requires `cols` and `lag`. |
| `lag_range` | a lag range such as `close_lag_1` through `close_lag_3` | Requires `col`, `start`, and `end`. |
| `lag_range_cols` | a lag range for each listed column | Requires `cols`, `start`, and `end`. |

## Stationarity And Long-Memory Helpers

These helpers are useful when you want to reduce non-stationarity while preserving more long-memory structure than a simple first difference would.

| Function | Adds or returns | Notes |
|---|---|---|
| `fractional_diff` | one `*_fracdiff` column per selected input column | Applies fixed-width fractional differentiation. Original columns are preserved. |
| `find_min_d` | scalar `d` value | Iterates over candidate orders and uses the Augmented Dickey-Fuller test to find the smallest stationary order. |

Two practical details matter:

- `fractional_diff` needs `cols=[...]` and writes new columns such as `close_fracdiff`.
- if one split is too short to produce the same fractional-diff column as another split, Manifest now drops that extra column during split alignment so the final `data_dict` stays consistent.

## Trade-Shape And Microstructure Features

These helpers need richer data than ordinary OHLCV bars.

| Function | Adds by default | Notes |
|---|---|---|
| `kline_imbalance` | `imbalance` | Requires `maker_ratio` and `no_of_trades`. Useful when those were carried through data retrieval or bar formation. |
| `conserved_flux_renormalization` | synthetic OHLCV plus `value_sum`, `vwap`, `flux_rel_std_mean`, `flux_rel_std_var`, `entropy_mean`, `entropy_var`, `Δflux_rms`, `Δentropy_rms` | Works on trade-level `datetime`, `price`, and `quantity`, then rolls those into kline-aligned diagnostics. |

## Choosing Between Indicators And Features

- Use an indicator when you want a direct market calculation such as RSI, ATR, or MACD.
- Use a feature when you want structure around those signals, such as lags, regimes, relative position, or multi-step aggregation.
- Use the lag helpers when the main value is temporal context rather than a new market calculation.
- Use `fractional_diff` when stationarity itself is part of the design problem, not just a preprocessing afterthought.

## Read Next

- [Indicators](Indicators.md) for lower-level signal primitives
- [Transforms](Transforms.md) for target shaping and post-model calibration helpers
- [Experiment Manifest](Experiment-Manifest.md) for how features plug into the split-first manifest pipeline
- [Utilities](Utilities.md) for the exported `adf_test()` helper that `find_min_d` builds on
