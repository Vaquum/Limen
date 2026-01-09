# Historical Data

The endpoints available through `limen.HistoricalData` provide rich and somewhat immediate access to Binance spot and futures data from year 2019 onwards, both kline and trade level data. Kline data is available at 1 second resolution for both spot and futures, and trade data is available at order and trade level. 

All of the endpoints are found in [`limen/historical_data.py`](../limen/historical_data.py).  

All of the endpoints rely on [Binance Market Data](https://data.binance.vision/?prefix=) as their source.

There are in total six distinct data endpoints:

- `HistoricalData.get_binance_file`
- `HistoricalData.get_spot_klines` (for both spot and futures)
- `HistoricalData.get_spot_trades`
- `HistoricalData.get_spot_agg_trades`
- `HistoricalData.get_futures_klines`
- `HistoricalData.get_futures_trades`
- `HistoricalData._get_data_for_test` (internal testing only)

All of these endpoints are available in the following manner: 

```
# Initialize the API
import limen
historical = limen.HistoricalData()

# Call one of the endpoints:
historical.get_spot_klines()

# Access the data
historical.data
```

All of the endpoints return `pl.DataFrame`. 

**NOTE:** There is an important performance cost that one suffers from moving the data out of the `pl.DataFrame` (e.g. to `pd.DataFrame`). This performance cost matters more for trades data, and for kline data data where `kline_size` is seconds and not minutes or longer.

## `HistoricalData.get_binance_file`

Get historical data from a Binance file based on the file URL. 

### Args

| Parameter    | Type        | Description                              |
|--------------|-------------|------------------------------------------|
| `file_url`   | `str`       | The URL of the Binance file.             |
| `has_header` | `bool`      | Whether the file has a header.           |
| `columns`    | `List[str]` | The columns to be included in the data.  |

### Returns

`self.data` (pl.DataFrame) with the columns being a result of the file that is read.

## `HistoricalData.get_spot_klines`

Get historical klines data for Binance spot.

### Args

| Parameter          | Type    | Description                              |
|--------------------|---------|------------------------------------------|
| `n_rows`           | `int`   | Number of rows to be pulled.             |
| `kline_size`       | `int`   | Size of the kline in seconds.            |
| `start_date_limit` | `str`   | The start date of the klines data.       |

### Returns

`self.data` (pl.DataFrame)

| Column Name        | Type        | Description                                                          |
|--------------------|-------------|----------------------------------------------------------------------|
| `datetime`         | `datetime`  | Start time of the kline period.                                      |
| `open`             | `float`     | Opening price of the kline period.                                   |
| `high`             | `float`     | Highest price reached during the kline period.                       |
| `low`              | `float`     | Lowest price reached during the kline period.                        |
| `close`            | `float`     | Closing price at the end of the kline period.                        |
| `mean`             | `float`     | Average price over the kline period.                                 |
| `std`              | `float`     | Standard deviation of price over the kline period.                   |
| `median`           | `float`     | Median price over the kline period.                                  |
| `iqr`              | `float`     | Interquartile range of prices during the kline period.               |
| `volume`           | `float`     | Total traded volume during the kline period.                         |
| `maker_ratio`      | `float`     | Ratio of maker-initiated trades to total trades in the period.       |
| `no_of_trades`     | `int`       | Number of trades executed during the kline period.                   |
| `open_liquidity`   | `float`     | Available liquidity at the open price.                               |
| `high_liquidity`   | `float`     | Available liquidity at the highest price.                            |
| `low_liquidity`    | `float`     | Available liquidity at the lowest price.                             |
| `close_liquidity`  | `float`     | Available liquidity at the close price.                              |
| `liquidity_sum`    | `float`     | Sum of liquidity across all price levels during the period.          |
| `maker_volume`     | `float`     | Sum of maker initiated traded volume in the period.                  |
| `maker_liquidity`  | `float`     | Sum of maker initiated liquidity in the period.                      |  

## `HistoricalData.get_spot_trades`

Get historical trades data for Binance spot.

### Args

| Parameter               | Type     | Description                                         |
|-------------------------|----------|-----------------------------------------------------|
| `month_year`            | `Tuple`  | The month of data to be pulled, e.g. `(3, 2025)`.   |
| `n_rows`                | `int`    | Number of latest rows to be pulled.                 |
| `n_random`              | `int`    | Number of random rows to be pulled.                 |
| `include_datetime_col`  | `bool`   | If the datetime column is to be included.           |
| `show_summary`          | `bool`   | Print query execution summary.                      |

### Returns

`self.data` (pl.DataFrame)

| Column Name        | Type        | Description                                        |
|--------------------|-------------|----------------------------------------------------|
| `trade_id`         | `int`       | Unique ID of the individual trade.                 |
| `timestamp`        | `int`       | Unix timestamp (ms) when the trade was executed.   |
| `price`            | `float`     | Price at which the trade occurred.                 |
| `quantity`         | `float`     | Quantity of the asset that was traded.             |
| `is_buyer_maker`   | `bool`      | `True` if the buyer was the market maker.          |
| `datetime`         | `datetime`  | Human-readable date/time derived from `timestamp`. |


## `HistoricalData.get_spot_agg_trades`

Get historical aggTrades data for Binance spot.

### Args

| Parameter               | Type     | Description                                         |
|-------------------------|----------|-----------------------------------------------------|
| `month_year`            | `Tuple`  | The month of data to be pulled, e.g. `(3, 2025)`.   |
| `n_rows`                | `int`    | Number of latest rows to be pulled.                 |
| `n_random`              | `int`    | Number of random rows to be pulled.                 |
| `include_datetime_col`  | `bool`   | If the datetime column is to be included.           |
| `show_summary`          | `bool`   | Print query execution summary.                      |

### Returns

`self.data` (pl.DataFrame)

| Column Name        | Type        | Description                                                                    |
|--------------------|-------------|--------------------------------------------------------------------------------|
| `agg_trade_id`     | `int`       | The ID of this aggregate trade, grouping together multiple individual orders.  |
| `timestamp`        | `int`       | Unix timestamp (ms) when these orders were executed.                           |
| `price`            | `float`     | Execution price shared by all orders in the aggregate.                         |
| `quantity`         | `float`     | Total quantity summed across all orders in this aggregate trade.               |
| `is_buyer_maker`   | `bool`      | `True` if the maker in this aggregate was the buyer (i.e. buyer placed the order). |
| `first_trade_id`   | `int`       | The individual order ID of the first order in this aggregate.                  |
| `last_trade_id`    | `int`       | The individual order ID of the last order in this aggregate.                   |
| `datetime`         | `datetime`  | Human-readable date/time corresponding to `timestamp`.                         |

## `HistoricalData.get_futures_klines`

Get historical klines data for Binance futures.

### Args

| Parameter          | Type    | Description                              |
|--------------------|---------|------------------------------------------|
| `n_rows`           | `int`   | Number of rows to be pulled.             |
| `kline_size`       | `int`   | Size of the kline in seconds.            |
| `start_date_limit` | `str`   | The start date of the klines data.       |

### Returns

`self.data` (pl.DataFrame)

| Column Name        | Type        | Description                                                          |
|--------------------|-------------|----------------------------------------------------------------------|
| `datetime`         | `datetime`  | Start time of the kline period.                                      |
| `open`             | `float`     | Opening price of the kline period.                                   |
| `high`             | `float`     | Highest price reached during the kline period.                       |
| `low`              | `float`     | Lowest price reached during the kline period.                        |
| `close`            | `float`     | Closing price at the end of the kline period.                        |
| `mean`             | `float`     | Average price over the kline period.                                 |
| `std`              | `float`     | Standard deviation of price over the kline period.                   |
| `median`           | `float`     | Median price over the kline period.                                  |
| `iqr`              | `float`     | Interquartile range of prices during the kline period.               |
| `volume`           | `float`     | Total traded volume during the kline period.                         |
| `maker_ratio`      | `float`     | Ratio of maker-initiated trades to total trades in the period.       |
| `no_of_trades`     | `int`       | Number of trades executed during the kline period.                   |
| `open_liquidity`   | `float`     | Available liquidity at the open price.                               |
| `high_liquidity`   | `float`     | Available liquidity at the highest price.                            |
| `low_liquidity`    | `float`     | Available liquidity at the lowest price.                             |
| `close_liquidity`  | `float`     | Available liquidity at the close price.                              |
| `liquidity_sum`    | `float`     | Sum of liquidity across all price levels during the period.          |
| `maker_volume`     | `float`     | Sum of maker initiated traded volume in the period.                  |
| `maker_liquidity`  | `float`     | Sum of maker initiated liquidity in the period.                      |  

## `HistoricalData.get_futures_trades`

Get historical trades data for Binance futures.

### Args

| Parameter               | Type                       | Description                                         |
|-------------------------|----------------------------|-----------------------------------------------------|
| `month_year`            | `tuple[int,int] \| None`   | (month, year) to fetch, e.g. `(3, 2025)`.           |
| `n_rows`                | `int \| None`              | If set, fetch this many latest rows instead.        |
| `n_random`              | `int \| None`              | If set, fetch this many random rows instead.        |
| `include_datetime_col`  | `bool`                     | Whether to include `datetime` in the result.        |
| `show_summary`          | `bool`                     | If a summary for the data is printed out.           |

### Returns

**NOTE:** In contrast to the other data endpoints, this one returns `pl.DataFrame` directly.

| Column Name        | Type        | Description                                        |
|--------------------|-------------|----------------------------------------------------|
| `futures_trade_id` | `int`       | Unique ID of the individual trade.                 |
| `timestamp`        | `int`       | Unix timestamp (ms) when the trade was executed.   |
| `price`            | `float`     | Price at which the trade occurred.                 |
| `quantity`         | `float`     | Quantity of the asset that was traded.             |
| `is_buyer_maker`   | `bool`      | `True` if the buyer was the market maker.          |
| `datetime`         | `datetime`  | Human-readable date/time derived from `timestamp`. |

## `HistoricalData._get_data_for_test`

Get test klines data from local CSV file for testing purposes.

**NOTE:** This is an internal test-only method used by SFMs to load sample data during test runs. It reads from `datasets/klines_2h_2020_2025.csv` which contains pre-downloaded klines data. This method is not intended for production use.

### Args

| Parameter  | Type          | Description                                                    |
|------------|---------------|----------------------------------------------------------------|
| `n_rows`   | `int \| None` | Number of rows to read from CSV (default: 5000). If `None`, reads entire file. |

### Returns

`self.data` (pl.DataFrame) - The structure matches the output from `get_spot_klines` or `get_futures_klines`, containing OHLCV data and other kline-related columns.

### Example Usage

```python
import limen
historical = limen.HistoricalData()

# Load 1000 rows of test data
historical._get_data_for_test(n_rows=1000)

# Access the data
historical.data
```