# Historical Data

The endpoints available through `loop.HistoricalData` provide rich and somewhat immediate access to Binance spot and futures data from year 2019 onwards, both kline and trade level data. Kline data is available at 1 second resolution for both spot and futures, and trade data is available at order and trade level. 

All of the endpoints are found in [`loop/data.py`](../loop/data.py).  

All of the endpoints rely on [Binance Market Data](https://data.binance.vision/?prefix=) as their source.

There are in total five distinct data endpoints:

- `HistoricalData.get_binance_file`
- `HistoricalData.get_historical_klines` (for both spot and futures)
- `HistoricalData.get_historical_trades`
- `HistoricalData.get_historical_agg_trades`
- `HistoricalData.get_historical_futures_trades`

All of these endpoints are available in the following manner: 

```
# Initialize the API
import loop
historical = loop.HistoricalData()

# Call one of the endpoints:
historical.get_historical_klines()

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

## `HistoricalData.get_historical_klines`

Get historical klines data for Binance spot or futures.

### Args

| Parameter          | Type    | Description                              |
|--------------------|---------|------------------------------------------|
| `n_rows`           | `int`   | Number of rows to be pulled.             |
| `kline_size`       | `int`   | Size of the kline in seconds.            |
| `start_date_limit` | `str`   | The start date of the klines data.       |
| `futures`          | `bool`  | If the data is from futures.             |

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
| `maker_ratio`      | `float`     | Ratio of maker-initiated volume to total volume in the period.       |
| `no_of_trades`     | `int`       | Number of trades executed during the kline period.                   |
| `open_liquidity`   | `float`     | Available liquidity at the open price.                               |
| `high_liquidity`   | `float`     | Available liquidity at the highest price.                            |
| `low_liquidity`    | `float`     | Available liquidity at the lowest price.                             |
| `close_liquidity`  | `float`     | Available liquidity at the close price.                              |
| `liquidity_sum`    | `float`     | Sum of liquidity across all price levels during the period.          |


## `HistoricalData.get_historical_trades`

Get historical trades data for Binance spot.

### Args

| Parameter               | Type     | Description                                         |
|-------------------------|----------|-----------------------------------------------------|
| `month_year`            | `Tuple`  | The month of data to be pulled, e.g. `(3, 2025)`.   |
| `n_latest`              | `int`    | Number of latest rows to be pulled.                 |
| `n_random`              | `int`    | Number of random rows to be pulled.                 |
| `include_datetime_col`  | `bool`   | If the datetime column is to be included.           |

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


## `HistoricalData.get_historical_agg_trades`

Get historical aggTrades data for Binance spot.

### Args

| Parameter               | Type     | Description                                         |
|-------------------------|----------|-----------------------------------------------------|
| `month_year`            | `Tuple`  | The month of data to be pulled, e.g. `(3, 2025)`.   |
| `n_rows`                | `int`    | Number of rows to be pulled.                        |
| `include_datetime_col`  | `bool`   | If the datetime column is to be included.           |

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


## `HistoricalData.get_historical_futures_trades`

Get historical trades data for Binance futures. 

### Args

| Parameter               | Type                       | Description                                         |
|-------------------------|----------------------------|-----------------------------------------------------|
| `month_year`            | `tuple[int,int] \| None`   | (month, year) to fetch, e.g. `(3, 2025)`.           |
| `n_rows`                | `int \| None`              | If set, fetch this many latest rows instead.        |
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