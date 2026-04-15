# Historical Data

`HistoricalData` is Limen's stateful file-backed data surface. It now has exactly three public retrieval methods:

- `get_spot_klines()`
- `get_binance_file()`
- `get_any_file()`

All three return `polars.DataFrame`, and each call also updates `historical.data` and `historical.data_columns`.

## How It Works

```python
import limen

historical = limen.HistoricalData()
data = historical.get_spot_klines(kline_size=3600, start_date_limit='2025-01-01')

assert data is historical.data
```

`HistoricalData` stays stateful for manifest compatibility, but the public methods now also return the loaded frame directly.

## Current Surface

| Method | Backend | Returns | Typical use |
|---|---|---|---|
| `get_spot_klines()` | latest Hugging Face BTCUSDT 1m parquet snapshot | BTCUSDT spot klines as `pl.DataFrame` | most common experiment input |
| `get_binance_file()` | direct Binance ZIP/CSV archive | normalized Binance file contents as `pl.DataFrame` | source-native Binance trade files |
| `get_any_file()` | local path or URL (`.parquet`, `.csv`, `.zip`) | loaded file contents as `pl.DataFrame` | test fixtures, local research files, remote datasets |

## `get_spot_klines()`

`get_spot_klines()` now reads from the BTCUSDT 1-minute dataset published at [vaquum/binance_btcusdt_1m_klines](https://huggingface.co/datasets/vaquum/binance_btcusdt_1m_klines).

By default it resolves the latest snapshot automatically, then aggregates upward from the file when you request a larger interval.

```python
from limen.data import HistoricalData

historical = HistoricalData()
data = historical.get_spot_klines(
    kline_size=3600,
    start_date_limit='2025-01-01',
)
```

Important rules:

- sub-1-minute klines are not supported
- `kline_size` must be a multiple of the source file interval
- the current Hugging Face dataset does not include `median` or `iqr`, so those columns are not returned

Returned columns:

- `datetime`, `open`, `high`, `low`, `close`
- `mean`, `std`
- `volume`, `maker_ratio`, `no_of_trades`
- `open_liquidity`, `high_liquidity`, `low_liquidity`, `close_liquidity`
- `liquidity_sum`, `maker_volume`, `maker_liquidity`

## `get_binance_file()`

`get_binance_file()` keeps the same role as before: load a Binance archive directly and normalize its `timestamp` / `datetime` columns.

```python
from limen.data import HistoricalData

historical = HistoricalData()
trades = historical.get_binance_file(
    file_url='https://data.binance.vision/data/spot/monthly/trades/BTCUSDT/BTCUSDT-trades-2025-01.zip',
    has_header=False,
    columns=[
        'trade_id', 'price', 'quantity', 'quote_qty',
        'timestamp', 'is_buyer_maker', 'is_best_match',
    ],
)
```

Use this when you want Binance source files rather than the curated kline dataset.

## `get_any_file()`

`get_any_file()` is the generic file ingestion path. It accepts a local path or URL and currently supports:

- `.parquet`
- `.csv`
- `.zip`

```python
from limen.data import HistoricalData

historical = HistoricalData()
data = historical.get_any_file(
    file_path_or_url=str(HistoricalData.DEFAULT_TEST_FILE_PATH),
    n_rows=5000,
)
```

It is the right choice for:

- local fixtures in tests
- repo-hosted CSV files
- remote parquet snapshots
- manifest test data sources

## Manifest Integration

Most manifest-driven experiments should now use:

- `HistoricalData.get_spot_klines` for production data
- `HistoricalData.get_spot_klines` with a smaller `n_rows` and coarser `kline_size` for lightweight test runs
- `HistoricalData.get_any_file` only when you intentionally want to load a specific local or remote file

```python
from limen.data import HistoricalData
from limen.experiment import Manifest

manifest = (
    Manifest()
    .set_data_source(
        method=HistoricalData.get_spot_klines,
        params={'kline_size': 3600, 'start_date_limit': '2025-01-01'},
    )
    .set_test_data_source(
        method=HistoricalData.get_spot_klines,
        params={'kline_size': 7200, 'n_rows': 5000},
    )
)
```

## Choosing The Right Surface

- Use `get_spot_klines()` for most Limen experiments and for manifest test sources that should stay on the public BTCUSDT path.
- Use `get_binance_file()` when you want direct Binance archives.
- Use `get_any_file()` for local fixtures, URLs, and generic file-backed ingestion.

## Read Next

- [Data Bars](Data-Bars.md)
- [Single File Decoder](Single-File-Decoder.md)
- [Experiment Manifest](Experiment-Manifest.md)
