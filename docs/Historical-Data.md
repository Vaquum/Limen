# Historical Data

`HistoricalData` is Limen's stateful data-access surface. It is the simplest way to get market data into Limen, whether you are fetching spot or futures data from the local ClickHouse-backed query layer, loading a Binance file directly, or using the bundled test dataset for manifest-driven examples.

Use this page when you need to answer three questions:

- which retrieval surface fits your experiment
- what columns that surface gives you
- how the retrieved data flows into a manifest or a `UniversalExperimentLoop`

## How `HistoricalData` Works

`HistoricalData` is stateful. Each retrieval method populates `historical.data` and updates `historical.data_columns`.

```python
import limen

historical = limen.HistoricalData()
historical.get_spot_klines(kline_size=3600, start_date_limit='2025-01-01')

data = historical.data
columns = historical.data_columns
```

This matters because the methods do not return a long-lived query object. They mutate the `HistoricalData` instance, and the rest of your code reads from that state.

## Current Surface

| Method | Backend | Returns in `historical.data` | Typical use |
|---|---|---|---|
| `get_binance_file()` | direct Binance file download | raw file contents normalized into a `pl.DataFrame` | one-off loading from Binance data files |
| `get_spot_klines()` | local ClickHouse query | aggregated spot OHLC-style klines | most common experiment input |
| `get_futures_klines()` | local ClickHouse query | aggregated futures OHLC-style klines | futures-side signal research |
| `get_spot_trades()` | local ClickHouse query | raw spot trades | trade-level analysis and custom prep |
| `get_spot_agg_trades()` | local ClickHouse query | raw spot aggregate trades | aggregated trade research |
| `get_futures_trades()` | local ClickHouse query | raw futures trades | futures microstructure research |
| `_get_data_for_test()` | local CSV bundled with the repo | sample kline dataset | tests, examples, and local dry runs |

## Access Patterns

### Kline Retrieval

`get_spot_klines()` and `get_futures_klines()` are the most common entry points for Limen experiments. They aggregate raw trades into time-based klines inside the query layer.

```python
from limen.data import HistoricalData

historical = HistoricalData()
historical.get_spot_klines(
    kline_size=3600,
    start_date_limit='2025-01-01',
)

data = historical.data
```

The returned dataframe contains the columns Limen's built-in manifests usually expect:

- `datetime`, `open`, `high`, `low`, `close`
- `mean`, `std`, `median`, `iqr`
- `volume`, `maker_ratio`, `no_of_trades`
- `open_liquidity`, `high_liquidity`, `low_liquidity`, `close_liquidity`
- `liquidity_sum`, `maker_volume`, `maker_liquidity`

Use kline retrieval when you want the standard Limen workflow: indicators, features, target shaping, backtests, and optional bar formation.

### Trade Retrieval

The trade endpoints expose raw rows instead of aggregated bars.

```python
from limen.data import HistoricalData

historical = HistoricalData(auth_token='your-clickhouse-password')
historical.get_spot_trades(month_year=(3, 2025))

trades = historical.data
```

For `get_spot_trades()`, `get_spot_agg_trades()`, and `get_futures_trades()`, exactly one of these inputs must be provided:

- `month_year`
- `n_rows`
- `n_random`

That contract comes from the underlying query helper and Limen will raise if more than one, or none, are supplied.

Typical output columns are:

- `get_spot_trades()`: `trade_id`, `timestamp`, `price`, `quantity`, `is_buyer_maker`, optional `datetime`
- `get_spot_agg_trades()`: `agg_trade_id`, `timestamp`, `price`, `quantity`, `is_buyer_maker`, `first_trade_id`, `last_trade_id`, optional `datetime`
- `get_futures_trades()`: `futures_trade_id`, `timestamp`, `price`, `quantity`, `is_buyer_maker`, optional `datetime`

Use the trade endpoints when you need custom preparation logic, custom aggregation, or non-kline research surfaces.

### Direct Binance File Loading

`get_binance_file()` loads a Binance CSV or ZIP file directly from a URL.

```python
from limen.data import HistoricalData

historical = HistoricalData()
historical.get_binance_file(
    file_url='https://data.binance.vision/data/spot/monthly/trades/BTCUSDT/BTCUSDT-trades-2025-01.zip',
    has_header=False,
    columns=['trade_id', 'price', 'quantity', 'quote_qty', 'timestamp', 'is_buyer_maker', 'is_best_match'],
)
```

This path is useful when you want source-native Binance files instead of the local query layer. Limen normalizes `timestamp` and adds `datetime`.

### Bundled Test Data

`_get_data_for_test()` is the test-only helper used throughout Limen's manifests and tests.

```python
from limen.data import HistoricalData

historical = HistoricalData()
historical._get_data_for_test(n_rows=5000)

data = historical.data
```

This reads from `datasets/klines_2h_2020_2025.csv`. It is the most convenient source for local examples, docs work, and quick smoke tests because it does not require ClickHouse access.

## Backend Notes

- `get_spot_klines()`, `get_futures_klines()`, `get_spot_trades()`, `get_spot_agg_trades()`, and `get_futures_trades()` use local ClickHouse-backed helpers.
- `auth_token` is passed through as the ClickHouse password.
- `get_binance_file()` does not depend on ClickHouse.
- `_get_data_for_test()` is local and deterministic enough for repeatable docs examples.

If you are writing public examples, prefer `_get_data_for_test()` or an explicit CSV load unless the point of the example is the ClickHouse query surface itself.

## Manifest Integration

Most manifest-driven experiments reference `HistoricalData` methods directly instead of fetching data outside the manifest.

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
        method=HistoricalData._get_data_for_test,
        params={'n_rows': 5000},
    )
)
```

In that flow:

- `LOOP_ENV='test'` uses the test data source when one is configured
- any other `LOOP_ENV` value uses the production data source
- `UniversalExperimentLoop` fetches the data automatically when you pass a manifest-driven SFD and do not pass `data=` explicitly

## Choosing The Right Surface

- Use kline methods for most Limen experiments.
- Use trade methods when you need custom aggregation or trade-level features.
- Use `get_binance_file()` when you want direct Binance source files.
- Use `_get_data_for_test()` for examples, docs, and local smoke tests.

## Read Next

- Continue to [Data Bars](Data-Bars.md) if you want to reshape kline data into threshold bars before feature engineering.
- Continue to [Single File Decoder](Single-File-Decoder.md) to package the experiment logic that will consume this data.
- Continue to [Experiment Manifest](Experiment-Manifest.md) to configure data fetching declaratively inside an SFD.
