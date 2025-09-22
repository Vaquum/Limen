# Bars

Alternative bar construction methods derived from klines data for enhanced sampling in data modeling for Loop.

## Purpose

Standard time-based kline bars sample data at fixed time intervals, which may not capture market microstructure effectively. Alternative bar construction methods sample based on market activity, providing more uniform statistical properties and better signal extraction for machine learning models.

## Bar Categories

| Category | Sampling Method | Use Cases |
|----------|-----------------|-----------|
| **Standard Bars** | Fixed thresholds on volume, trades, or liquidity | Basic activity-based sampling, regime-neutral feature engineering |
| **Imbalance Bars** | Directional imbalance accumulation with adaptive thresholds | Detecting shifts in buying/selling pressure, momentum strategies |
| **Run Bars** | Consecutive sequences of same-direction activity | Identifying trend continuation and exhaustion points |

## Standard Bar Columns

All standard bar functions return a `pl.DataFrame` with the following structure:

| Column Name | Type | Description |
|-------------|------|-------------|
| `datetime` | `datetime` | Start time of the bar period |
| `open` | `float` | Opening price of the bar period |
| `high` | `float` | Highest price reached during the bar period |
| `low` | `float` | Lowest price reached during the bar period |
| `close` | `float` | Closing price at the end of the bar period |
| `mean` | `float` | Average price at the end of the bar period |
| `volume` | `float` | Total volume accumulated in the bar |
| `no_of_trades` | `int` | Number of trades accumulated in the bar |
| `liquidity_sum` | `float` | Total liquidity accumulated in the bar |
| `maker_ratio` | `float` | Proportion of trades executed by makers (0.0 to 1.0) |
| `maker_volume` | `float` | Total volume executed by makers |
| `maker_liquidity` | `float` | Total liquidity provided by makers |
| `bar_count` | `int` | Number of base klines aggregated into this bar |
| `base_interval` | `float` | Original kline interval in seconds (e.g., 7200.0 for 2h) |

## `loop.data.bars`

### `volume_bars`

Compute volume bars with fixed volume size sampling.

#### Args

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `pl.DataFrame` | Klines dataframe |
| `volume_threshold` | `float` | Volume threshold per bar |

#### Returns

`pl.DataFrame`: Standard bar columns as defined above.

### `trade_bars`

Compute trade bars with fixed trade count sampling.

#### Args

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `pl.DataFrame` | Klines dataframe |
| `trade_threshold` | `int` | Number of trades per bar |

#### Returns

`pl.DataFrame`: Standard bar columns as defined above.

### `liquidity_bars`

Compute liquidity bars with fixed liquidity sampling.

#### Args

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `pl.DataFrame` | Klines dataframe |
| `liquidity_threshold` | `float` | Liquidity threshold per bar |

#### Returns

`pl.DataFrame`: Standard bar columns as defined above.

## Imbalance Bars (TBA)

## Run Bars (TBA)
