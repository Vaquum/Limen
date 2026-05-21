# Historical Data

`HistoricalData` is Limen's stateful file-backed data surface. It has four public retrieval methods:

- `get_spot_klines()`
- `get_spot_dollar_klines()`
- `get_binance_file()`
- `get_any_file()`

All four return `polars.DataFrame`, and each call also updates `historical.data` and `historical.data_columns`.

## How It Works

```python
import limen

historical = limen.HistoricalData()
data = historical.get_spot_klines(kline_size=3600, start_date_limit='2025-01-01')

assert data is historical.data
```

`HistoricalData` stays stateful for manifest compatibility, and the public methods also return the loaded frame directly.

## Current Surface

| Method | Backend | Returns | Typical use |
|---|---|---|---|
| `get_spot_klines()` | Hugging Face BTCUSDT parquet datasets | BTCUSDT spot klines as `pl.DataFrame` | most common experiment input |
| `get_spot_dollar_klines()` | Hugging Face BTCUSDT dollar-bar parquet datasets | BTCUSDT spot dollar bars as `pl.DataFrame` | event-time experiments |
| `get_binance_file()` | direct Binance ZIP/CSV archive | normalized Binance file contents as `pl.DataFrame` | source-native Binance trade files |
| `get_any_file()` | local path or URL (`.parquet`, `.csv`, `.zip`) | loaded file contents as `pl.DataFrame` | test fixtures, local research files, remote datasets |

## `get_spot_klines()`

`get_spot_klines()` reads from the BTCUSDT datasets published on Hugging Face.

By default it reads native [1m](https://huggingface.co/datasets/vaquum/binance_btcusdt_1m_klines), [15m](https://huggingface.co/datasets/vaquum/binance_btcusdt_15m_klines), [30m](https://huggingface.co/datasets/vaquum/binance_btcusdt_30m_klines), [1h](https://huggingface.co/datasets/vaquum/binance_btcusdt_1h_klines), [2h](https://huggingface.co/datasets/vaquum/binance_btcusdt_2h_klines), or [4h](https://huggingface.co/datasets/vaquum/binance_btcusdt_4h_klines) datasets for matching `kline_size` values, and otherwise the 1m dataset before aggregating upward.

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
- `row_count_limit` returns the latest rows after date filtering and aggregation
- date-only `end_date_limit` values include the full named day
- `start_date_limit` and `end_date_limit` may define a closed window only when `row_count_limit` is unset

Returned columns:

- `datetime`, `open`, `high`, `low`, `close`
- `mean`, `std`
- `volume`, `maker_ratio`, `no_of_trades`
- `open_liquidity`, `high_liquidity`, `low_liquidity`, `close_liquidity`
- `liquidity_sum`, `maker_volume`, `maker_liquidity`

## `get_spot_dollar_klines()`

`get_spot_dollar_klines()` reads BTCUSDT dollar bars from Vaquum Hugging Face datasets.

It reads native [1M](https://huggingface.co/datasets/vaquum/binance_btcusdt_1M_dollar_klines), [15M](https://huggingface.co/datasets/vaquum/binance_btcusdt_15M_dollar_klines), [30M](https://huggingface.co/datasets/vaquum/binance_btcusdt_30M_dollar_klines), [60M](https://huggingface.co/datasets/vaquum/binance_btcusdt_60M_dollar_klines), [120M](https://huggingface.co/datasets/vaquum/binance_btcusdt_120M_dollar_klines), or [240M](https://huggingface.co/datasets/vaquum/binance_btcusdt_240M_dollar_klines) datasets for matching `dollar_bar_size` values. Other multiples use the largest available lower dollar-bar source and aggregate upward.

```python
from limen.data import HistoricalData

historical = HistoricalData()
data = historical.get_spot_dollar_klines(
    dollar_bar_size=30_000_000,
    start_date_limit='2025-01-01',
)
```

Important rules:

- `dollar_bar_size` must be a positive integer
- supported native sizes are resolved directly instead of deriving from the 1-minute kline file
- non-native sizes must be multiples of the selected source dollar-bar size
- returned `datetime` is the dollar bar start time
- returned columns match `get_spot_klines()` for manifest compatibility

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
    file_path_or_url='path/to/local/data.parquet',
    row_count_limit=5000,
)
```

It is the right choice for:

- local fixtures in tests
- repo-hosted CSV files
- remote parquet snapshots
- manifest test data sources

## Manifest Integration

Most manifest-driven experiments should use:

- `HistoricalData.get_spot_klines` for production data
- `HistoricalData.get_spot_dollar_klines` for event-time dollar-bar production data
- `HistoricalData.get_spot_klines` with a smaller `row_count_limit` and coarser `kline_size` for lightweight test runs
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
        params={'kline_size': 7200, 'row_count_limit': 5000},
    )
)
```

## Choosing The Right Surface

- Use `get_spot_klines()` for most Limen experiments and for manifest test sources that should stay on the public BTCUSDT path.
- Use `get_spot_dollar_klines()` when market activity, not wall-clock time, should define each row.
- Use `get_binance_file()` when you want direct Binance archives.
- Use `get_any_file()` for local fixtures, URLs, and generic file-backed ingestion.

## Read Next

- [Data Bars](Data-Bars.md)
- [Single File Decoder](Single-File-Decoder.md)
- [Experiment Manifest](Experiment-Manifest.md)
