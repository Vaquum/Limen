# Features

Features are more complex than Indicators, and often involve further refining Indicators or combining several Indicators into a single Feature.

## `loop.features`

### `atr_percent_sma`

Compute ATR as percentage of close price using Simple Moving Average.

#### Args

| Parameter  | Type             | Description                                        |
| ---------- | ---------------- | -------------------------------------------------- |
| `data`   | `pl.DataFrame` | Klines dataset with 'high', 'low', 'close' columns |
| `period` | `int`          | Number of periods for ATR calculation              |

#### Returns

`pl.DataFrame`: The input data with a new column 'atr_percent_sma'

### `atr_sma`

Compute Average True Range using Simple Moving Average.

NOTE: Different from standard ATR which uses Wilder's smoothing.

#### Args

| Parameter  | Type             | Description                                        |
| ---------- | ---------------- | -------------------------------------------------- |
| `data`   | `pl.DataFrame` | Klines dataset with 'high', 'low', 'close' columns |
| `period` | `int`          | Number of periods for ATR calculation              |

#### Returns

`pl.DataFrame`: The input data with a new column 'atr_sma'

### `breakout_features`

Compute comprehensive breakout-related features including lags, stats, and ROC.

#### Args

| Parameter     | Type             | Description                                            |
| ------------- | ---------------- | ------------------------------------------------------ |
| `data`      | `pl.DataFrame` | Klines dataset with breakout signal columns            |
| `long_col`  | `str`          | Column name for long breakout signals                  |
| `short_col` | `str`          | Column name for short breakout signals                 |
| `lookback`  | `int`          | Number of periods for feature calculation              |
| `horizon`   | `int`          | Number of periods to shift for avoiding lookahead bias |
| `target`    | `str`          | Target column name for filtering null values           |

#### Returns

`pl.DataFrame`: The input data with multiple breakout feature columns added

### `close_position`

Compute close position within the high-low range as percentage.

#### Args

| Parameter | Type             | Description                                            |
| --------- | ---------------- | ------------------------------------------------------ |
| `data`  | `pl.DataFrame` | Klines dataset with 'high', 'low', and 'close' columns |

#### Returns

`pl.DataFrame`: The input data with a new column 'close_position'

### `conserved_flux_renormalization`

Compute multi-scale, conserved-flux features and their deviation scores for each k-line—turning raw trade ticks into a six-value fingerprint that flags hours where the dollar flow or trade-size entropy breaks scale-invariant behaviour.

Read more in: [Conserved Flux Renormalization](Conserved-Flux-Renormalization.md)

#### Args

| Parameter          | Type             | Description                      |
| ------------------ | ---------------- | -------------------------------- |
| `trades_df`      | `pl.DataFrame` | The trades DataFrame.            |
| `kline_interval` | `str`          | The kline interval.              |
| `base_window_s`  | `int`          | The base window size.            |
| `levels`         | `int`          | The number of levels to compute. |

#### Returns

`pl.DataFrame`: A klines DataFrame with the CFR features

| Column                | Type         | Brief description (1-line)                                          |
| --------------------- | ------------ | ------------------------------------------------------------------- |
| `datetime`          | datetime[ms] | Start timestamp of the k-line window (bucket-aligned).              |
| `open`              | float64      | First trade price in the window.                                    |
| `high`              | float64      | Highest trade price in the window.                                  |
| `low`               | float64      | Lowest trade price in the window.                                   |
| `close`             | float64      | Last trade price in the window.                                     |
| `volume`            | float64      | Total BTC traded in the window.                                     |
| `value_sum`         | float64      | Dollar notional traded ∑ (price × quantity).                      |
| `vwap`              | float64      | Volume-weighted average price (value_sum / volume).                 |
| `flux_rel_std_mean` | float64      | Mean relative σ/μ of value-flux across the 6 nested scales.       |
| `flux_rel_std_var`  | float64      | Variance of that σ/μ ladder (how one scale dominates).            |
| `entropy_mean`      | float64      | Mean Shannon entropy (bits) of trade-size mix across scales.        |
| `entropy_var`       | float64      | Variance of the entropy ladder (patchiness of size mix).            |
| `Δflux_rms`        | float64      | RMS gap from an ideal*flat* flux ladder (0 = scale-neutral flow). |
| `Δentropy_rms`     | float64      | RMS gap from a perfect 1-bit-per-octave entropy drop.               |

### `distance_from_high`

Compute distance from rolling high as percentage.

#### Args

| Parameter  | Type             | Description                                    |
| ---------- | ---------------- | ---------------------------------------------- |
| `data`   | `pl.DataFrame` | Klines dataset with 'high', 'close' columns    |
| `period` | `int`          | Number of periods for rolling high calculation |

#### Returns

`pl.DataFrame`: The input data with a new column 'distance_from_high'

### `distance_from_low`

Compute distance from rolling low as percentage.

#### Args

| Parameter  | Type             | Description                                   |
| ---------- | ---------------- | --------------------------------------------- |
| `data`   | `pl.DataFrame` | Klines dataset with 'low', 'close' columns    |
| `period` | `int`          | Number of periods for rolling low calculation |

#### Returns

`pl.DataFrame`: The input data with a new column 'distance_from_low'

### `ema_breakout`

Compute EMA breakout indicator based on price deviation from EMA.

#### Args

| Parameter            | Type             | Description                             |
| -------------------- | ---------------- | --------------------------------------- |
| `data`             | `pl.DataFrame` | Klines dataset with price columns       |
| `target_col`       | `str`          | Column name to analyze for breakouts    |
| `ema_span`         | `int`          | Period for EMA calculation              |
| `breakout_delta`   | `float`        | Threshold for breakout detection        |
| `breakout_horizon` | `int`          | Lookback period for breakout validation |

#### Returns

`pl.DataFrame`: The input data with a new column 'breakout_ema'

### `gap_high`

Compute gap between current high and previous close as percentage.

#### Args

| Parameter | Type             | Description                                    |
| --------- | ---------------- | ---------------------------------------------- |
| `data`  | `pl.DataFrame` | Klines dataset with 'high' and 'close' columns |

#### Returns

`pl.DataFrame`: The input data with a new column 'gap_high'

### `kline_imbalance`

Compute rolling buyer/seller imbalance over klines instead of raw trades.

#### Args

| Parameter  | Type             | Description                                      |
| ---------- | ---------------- | ------------------------------------------------ |
| `data`   | `pl.DataFrame` | Klines dataset with 'open' and 'close' columns   |
| `window` | `int`          | Number of periods for rolling window calculation |

#### Returns

`pl.DataFrame`: The input data with a new column 'kline_imbalance'

### Lagged Features

The lagged features module provides a consolidated set of functions for creating lagged versions of columns. All functions are implemented using efficient vectorized Polars expressions.

NOTE: All lag functions are available in `loop.features.lagged_features` and exported through `loop.features`.

#### `lag_range_cols`

Compute multiple lagged versions of multiple columns over a range.

This is the core function that all other lag functions derive from, using fully vectorized Polars expressions for maximum efficiency.

##### Args

| Parameter | Type             | Description                           |
| --------- | ---------------- | ------------------------------------- |
| `data`  | `pl.DataFrame` | Klines dataset with specified columns |
| `cols`  | `list[str]`    | The list of column names to lag       |
| `start` | `int`          | The start of lag range (inclusive)    |
| `end`   | `int`          | The end of lag range (inclusive)      |

##### Returns

`pl.DataFrame`: The input data with the lagged columns appended

#### `lag_range`

Compute multiple lagged versions of a column over a range.

##### Args

| Parameter | Type             | Description                          |
| --------- | ---------------- | ------------------------------------ |
| `data`  | `pl.DataFrame` | Klines dataset with specified column |
| `col`   | `str`          | The column name to lag               |
| `start` | `int`          | The start of lag range (inclusive)   |
| `end`   | `int`          | The end of lag range (inclusive)     |

##### Returns

`pl.DataFrame`: The input data with the lagged columns appended

#### `lag_columns`

Compute lagged versions of multiple columns.

##### Args

| Parameter | Type             | Description                           |
| --------- | ---------------- | ------------------------------------- |
| `data`  | `pl.DataFrame` | Klines dataset with specified columns |
| `cols`  | `list[str]`    | The list of column names to lag       |
| `lag`   | `int`          | The number of periods to lag          |

##### Returns

`pl.DataFrame`: The input data with the lagged columns appended

#### `lag_column`

Compute a lagged version of a column.

##### Args

| Parameter | Type              | Description                                       |
| --------- | ----------------- | ------------------------------------------------- |
| `data`  | `pl.DataFrame`  | Klines dataset with specified column              |
| `col`   | `str`           | The column name to lag                            |
| `lag`   | `int`           | The number of periods to lag                      |
| `alias` | `str, optional` | New column name. If None, uses alias f"lag_{lag}" |

##### Returns

`pl.DataFrame`: The input data with the lagged column appended

### `price_range_position`

Compute price position within rolling high-low range.

#### Args

| Parameter  | Type             | Description                                        |
| ---------- | ---------------- | -------------------------------------------------- |
| `data`   | `pl.DataFrame` | Klines dataset with 'high', 'low', 'close' columns |
| `period` | `int`          | Number of periods for rolling range calculation    |

#### Returns

`pl.DataFrame`: The input data with a new column 'price_range_position'

### `ma_slope_regime`

Compute regime using the slope of SMA(close, period) with optional normalization.

#### Args

| Parameter          | Type           | Description                                           |
|--------------------|----------------|-------------------------------------------------------|
| `data`             | `pl.DataFrame` | Klines dataset with 'close' column                    |
| `period`           | `int`          | SMA period for the slope                              |
| `threshold`        | `float`        | Slope threshold; applied after normalization when enabled |
| `normalize_by_std` | `bool`         | Whether to divide slope by rolling std(period)        |

#### Returns

`pl.DataFrame`: The input data with a new column 'regime_ma_slope'

### `price_vs_band_regime`

Compute regime by comparing 'close' to center ± k × band width over a rolling window.

#### Args

| Parameter | Type                          | Description                            |
|----------|--------------------------------|----------------------------------------|
| `data`   | `pl.DataFrame`                 | Klines dataset with 'close' column     |
| `period` | `int`                          | Rolling period for center and band width |
| `band`   | `Literal['std', 'dev_std']`    | Band width type to use                  |
| `k`      | `float`                        | Band multiplier applied to the width    |

#### Returns

`pl.DataFrame`: The input data with a new column 'regime_price_band'

### `breakout_percentile_regime`

Compute regime classification by percentile position of 'close' within rolling [low, high].

#### Args

| Parameter | Type           | Description                                  |
|----------|----------------|----------------------------------------------|
| `data`   | `pl.DataFrame` | Klines dataset with 'high', 'low', 'close' columns |
| `period` | `int`          | Rolling window for high/low range            |
| `p_hi`   | `float`        | Upper percentile threshold in [0, 1]         |
| `p_lo`   | `float`        | Lower percentile threshold in [0, 1]         |

#### Returns

`pl.DataFrame`: The input data with a new column 'regime_breakout_pct'

### `window_return_regime`

Compute regime using windowed return close/close.shift(period) - 1.

#### Args

| Parameter | Type           | Description                                 |
|----------|----------------|---------------------------------------------|
| `data`   | `pl.DataFrame` | Klines dataset with 'close' column          |
| `period` | `int`          | Window length for return calculation        |
| `r_hi`   | `float`        | Upper threshold for Up regime               |
| `r_lo`   | `float`        | Lower threshold for Down regime             |

#### Returns

`pl.DataFrame`: The input data with a new column 'regime_window_return'

### `hh_hl_structure_regime`

Compute regime by higher-high / higher-low market structure within a rolling window.

#### Args

| Parameter        | Type           | Description                                    |
|------------------|----------------|------------------------------------------------|
| `data`           | `pl.DataFrame` | Klines dataset with 'high', 'low' columns      |
| `window`         | `int`          | Rolling window size for structure count        |
| `score_threshold`| `int`          | Absolute score threshold for Up/Down classification |

#### Returns

`pl.DataFrame`: The input data with a new column 'regime_hh_hl'

### `quantile_flag`

Mark rows where `col` exceeds the (1 - q) quantile.

#### Args

| Parameter         | Type             | Description                                                                                                                                                                        |
| ----------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data`          | `pl.DataFrame` | Klines dataset with specified column                                                                                                                                               |
| `col`           | `str`          | The column name on which to compute the quantile.                                                                                                                                  |
| `q`             | `float`        | A value in [0,1]; if q = 0.1, use the 90% quantile.                                                                                                                                |
| `cutoff`        | `float`        | Optional pre-calculated cutoff value. If provided, this value is used instead of calculating from data. This prevents data leakage when applying training thresholds to test data. |
| `return_cutoff` | `bool`         | If True, returns a tuple (data, cutoff) instead of just the data. Useful for applying the same cutoff to multiple datasets.                                                        |

#### Returns

`pl.DataFrame`: The input data with a new UInt8 column "quantile_flag" that is 1 when `col` > cutoff, else 0
OR
tuple: (pl.DataFrame, float) if return_cutoff is True

### `range_pct`

Compute range as percentage of close price (high-low)/close.

#### Args

| Parameter | Type             | Description                                            |
| --------- | ---------------- | ------------------------------------------------------ |
| `data`  | `pl.DataFrame` | Klines dataset with 'high', 'low', and 'close' columns |

#### Returns

`pl.DataFrame`: The input data with a new column 'range_pct'

### `trend_strength`

Compute trend strength based on moving average divergence.

#### Args

| Parameter       | Type             | Description                                |
| --------------- | ---------------- | ------------------------------------------ |
| `data`        | `pl.DataFrame` | Klines dataset with 'close' column         |
| `fast_period` | `int`          | Number of periods for fast SMA calculation |
| `slow_period` | `int`          | Number of periods for slow SMA calculation |

#### Returns

`pl.DataFrame`: The input data with a new column 'trend_strength'

### `volume_regime`

Compute volume regime (current vs average volume).

#### Args

| Parameter    | Type             | Description                                      |
| ------------ | ---------------- | ------------------------------------------------ |
| `data`     | `pl.DataFrame` | Klines dataset with 'volume' column              |
| `lookback` | `int`          | Number of periods for volume average calculation |

#### Returns

`pl.DataFrame`: The input data with a new column 'volume_regime'

### `vwap`

Compute Volume Weighted Average Price (VWAP) for each kline over its trading day.

#### Args

| Parameter      | Type             | Description                                  |
| -------------- | ---------------- | -------------------------------------------- |
| `data`       | `pl.DataFrame` | Klines dataset with price and volume columns |
| `price_col`  | `str`          | Name of the price column                     |
| `volume_col` | `str`          | Name of the volume column                    |

#### Returns

`pl.DataFrame`: The input data with a new column 'vwap'

### `ichimoku_cloud`

Compute Ichimoku Cloud components for trend and momentum analysis.

#### Args

| Parameter | Type             | Description                                        |
| --------- | ---------------- | -------------------------------------------------- |
| `data`  | `pl.DataFrame` | Klines dataset with 'high', 'low', 'close' columns |
| `tenkan_period`  | `int` | Lookback period for Tenkan-sen |
| `kijun_period`  | `int` | Lookback period for Kijun-sen |
| `senkou_b_period`  | `int` | Lookback period for Senkou Span B |
| `displacement`  | `int` | Number of periods to shift Senkou spans and Chikou span |

#### Returns

`pl.DataFrame`: The input data with new columns: 'tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou'
