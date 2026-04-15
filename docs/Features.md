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
        .with_model(logreg_binary)
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
| `compute_quantile_cutoff` | scalar cutoff value | Utility helper, not a DataFrame transform. |
| `quantile_flag` | `quantile_flag` | Commonly used in targets after computing the cutoff on train only. |

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
- Use `compute_quantile_cutoff` and `quantile_flag` when the target boundary itself is part of the split-safe training logic.
- Use `fractional_diff` when stationarity itself is part of the design problem, not just a preprocessing afterthought.

## Read Next

- [Indicators](Indicators.md) for lower-level signal primitives
- [Transforms](Transforms.md) for target shaping and post-model calibration helpers
- [Experiment Manifest](Experiment-Manifest.md) for how features plug into the split-first manifest pipeline
- [Utilities](Utilities.md) for the exported `adf_test()` helper that `find_min_d` builds on
