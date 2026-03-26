# Bars

Limen currently supports threshold-based bar formation over existing kline data. These functions aggregate consecutive kline rows until a fixed activity threshold is reached, producing alternative OHLCV bars for experiment input.

## Current Scope

The implemented bar surface today is:

- Volume bars
- Trade bars
- Liquidity bars

These are all built on top of kline data that already contains the columns needed for aggregation, such as `volume`, `no_of_trades`, and `liquidity_sum`.

## Shared Output Schema

All supported bar functions return a `pl.DataFrame` with the following columns:

| Column Name | Type | Description |
|-------------|------|-------------|
| `datetime` | `datetime` | Start time of the aggregated bar |
| `open` | `float` | Opening price of the bar |
| `high` | `float` | Highest price reached in the bar |
| `low` | `float` | Lowest price reached in the bar |
| `close` | `float` | Closing price of the bar |
| `volume` | `float` | Total volume accumulated in the bar |
| `no_of_trades` | `int` | Total trade count accumulated in the bar |
| `liquidity_sum` | `float` | Total liquidity accumulated in the bar |
| `maker_ratio` | `float` | Trade-count-weighted maker ratio inside the bar |
| `maker_volume` | `float` | Total maker volume accumulated in the bar |
| `maker_liquidity` | `float` | Total maker liquidity accumulated in the bar |
| `mean` | `float` | Trade-count-weighted mean price inside the bar |
| `bar_count` | `int` | Number of source klines merged into the bar |
| `base_interval` | `float` | Interval of the source kline series in seconds |

## `limen.data.bars`

### `volume_bars`

Aggregate rows until cumulative volume reaches `volume_threshold`.

#### Args

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `pl.DataFrame` | Klines dataframe |
| `volume_threshold` | `float` | Volume threshold per bar |

#### Returns

`pl.DataFrame`: Bar dataframe using the shared output schema above.

### `trade_bars`

Aggregate rows until cumulative trade count reaches `trade_threshold`.

#### Args

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `pl.DataFrame` | Klines dataframe |
| `trade_threshold` | `int` | Trade-count threshold per bar |

#### Returns

`pl.DataFrame`: Bar dataframe using the shared output schema above.

### `liquidity_bars`

Aggregate rows until cumulative liquidity reaches `liquidity_threshold`.

#### Args

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `pl.DataFrame` | Klines dataframe |
| `liquidity_threshold` | `float` | Liquidity threshold per bar |

#### Returns

`pl.DataFrame`: Bar dataframe using the shared output schema above.

## Manifest Integration

Use `Manifest.set_bar_formation()` to apply these bars inside a manifest-driven experiment:

```python
from limen.data.utils import compute_data_bars

# In params()
# {'bar_type': ['volume'], 'volume_threshold': [50_000, 100_000]}

manifest.set_bar_formation(
    compute_data_bars,
    bar_type='bar_type',
    volume_threshold='volume_threshold',
)
```

`bar_type` must be present in `round_params` for the bar-formation step to run, so it should be included in `params()`. String values such as `bar_type='bar_type'` and `volume_threshold='volume_threshold'` follow Limen's standard "resolve this from `round_params`" convention.

`compute_data_bars()` currently supports `bar_type` values `base`, `trade`, `volume`, and `liquidity`.
