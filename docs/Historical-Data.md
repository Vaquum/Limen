# Historical Data

`HistoricalData` is Limen's stateful data-access surface for Binance market data. In practice, the class mixes two access patterns:

- direct Binance-file loading through `get_binance_file()`
- ClickHouse-backed kline and trade queries through the remaining `get_*` methods

The implementation lives in [`limen/data/historical_data.py`](../limen/data/historical_data.py).

## Usage Pattern

All methods populate `historical.data` with a `pl.DataFrame` and update `historical.data_columns`.

```python
import limen

historical = limen.HistoricalData()

historical.get_spot_klines(kline_size=3600, start_date_limit='2025-01-01')

data = historical.data
```

## Current Surface

The class currently exposes six retrieval methods plus one internal test helper:

- `HistoricalData.get_binance_file`
- `HistoricalData.get_spot_klines`
- `HistoricalData.get_futures_klines`
- `HistoricalData.get_spot_trades`
- `HistoricalData.get_spot_agg_trades`
- `HistoricalData.get_futures_trades`
- `HistoricalData._get_data_for_test` (internal testing helper)

## Backend Notes

- `get_spot_klines`, `get_futures_klines`, `get_spot_trades`, `get_spot_agg_trades`, and `get_futures_trades` query local ClickHouse-backed tables through Limen's internal query helpers.
- `get_binance_file` loads a Binance file directly from a URL.
- `_get_data_for_test` loads a local CSV bundled for testing.
- `auth_token` is forwarded to the ClickHouse query layer for authenticated local access.

## `HistoricalData.get_binance_file`

Load a Binance CSV or ZIP file directly from a URL.

### Args

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_url` | `str` | URL of the Binance file |
| `has_header` | `bool` | Whether the file already includes a header row |
| `columns` | `list[str] \| None` | Column names for headerless files |

### Behavior

- Populates `self.data`
- Normalizes `timestamp`
- Adds a `datetime` column derived from `timestamp`

## `HistoricalData.get_spot_klines`

Query aggregated spot-market klines.

### Args

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_rows` | `int \| None` | Optional limit on returned rows |
| `kline_size` | `int` | Kline size in seconds |
| `start_date_limit` | `str \| None` | Optional lower bound for `datetime` |

### Returns

Populates `self.data` with kline rows containing:

- `datetime`, `open`, `high`, `low`, `close`
- `mean`, `std`, `median`, `iqr`
- `volume`, `maker_ratio`, `no_of_trades`
- `open_liquidity`, `high_liquidity`, `low_liquidity`, `close_liquidity`
- `liquidity_sum`, `maker_volume`, `maker_liquidity`

## `HistoricalData.get_futures_klines`

Same surface as `get_spot_klines`, but sourced from the futures trade table.

### Args

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_rows` | `int \| None` | Optional limit on returned rows |
| `kline_size` | `int` | Kline size in seconds |
| `start_date_limit` | `str \| None` | Optional lower bound for `datetime` |

## `HistoricalData.get_spot_trades`

Query raw spot trades.

Exactly one of `month_year`, `n_rows`, or `n_random` should be provided.

### Args

| Parameter | Type | Description |
|-----------|------|-------------|
| `month_year` | `tuple[int, int] \| None` | Month/year filter such as `(3, 2025)` |
| `n_rows` | `int \| None` | Latest-row limit |
| `n_random` | `int \| None` | Random-row limit |
| `include_datetime_col` | `bool` | Whether to include `datetime` |
| `show_summary` | `bool` | Whether to print query summary |

### Returns

Populates `self.data` with:

- `trade_id`
- `timestamp`
- `price`
- `quantity`
- `is_buyer_maker`
- `datetime` when requested

## `HistoricalData.get_spot_agg_trades`

Query raw spot aggregate trades.

Exactly one of `month_year`, `n_rows`, or `n_random` should be provided.

### Args

| Parameter | Type | Description |
|-----------|------|-------------|
| `month_year` | `tuple[int, int] \| None` | Month/year filter such as `(3, 2025)` |
| `n_rows` | `int \| None` | Latest-row limit |
| `n_random` | `int \| None` | Random-row limit |
| `include_datetime_col` | `bool` | Whether to include `datetime` |
| `show_summary` | `bool` | Whether to print query summary |

### Returns

Populates `self.data` with:

- `agg_trade_id`
- `timestamp`
- `price`
- `quantity`
- `is_buyer_maker`
- `first_trade_id`
- `last_trade_id`
- `datetime` when requested

## `HistoricalData.get_futures_trades`

Query raw futures trades.

Exactly one of `month_year`, `n_rows`, or `n_random` should be provided.

### Args

| Parameter | Type | Description |
|-----------|------|-------------|
| `month_year` | `tuple[int, int] \| None` | Month/year filter such as `(3, 2025)` |
| `n_rows` | `int \| None` | Latest-row limit |
| `n_random` | `int \| None` | Random-row limit |
| `include_datetime_col` | `bool` | Whether to include `datetime` |
| `show_summary` | `bool` | Whether to print query summary |

### Returns

In the current implementation this behaves like the other query methods: it populates `self.data` with:

- `futures_trade_id`
- `timestamp`
- `price`
- `quantity`
- `is_buyer_maker`
- `datetime` when requested

## `HistoricalData._get_data_for_test`

Internal helper that loads test klines from `datasets/klines_2h_2020_2025.csv`.

### Args

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_rows` | `int \| None` | Number of rows to read, or `None` for the full file |

### Returns

Populates `self.data` with a local kline sample used by tests and manifest test-mode flows.

## Manifest Usage

Manifest-driven SFDs typically reference these methods directly:

```python
from limen.data import HistoricalData
from limen.experiment import Manifest

manifest = (Manifest()
    .set_data_source(
        method=HistoricalData.get_spot_klines,
        params={'kline_size': 3600, 'start_date_limit': '2025-01-01'},
    )
    .set_test_data_source(method=HistoricalData._get_data_for_test)
)
```
