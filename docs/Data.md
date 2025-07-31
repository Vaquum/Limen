# Data

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

**NOTE:** There is an important performance cost that one suffers from moving the data out of the `pl.DataFrame` (e.g. to `pd.DataFrame`). 

## `HistoricalData.get_binance_file`

Get historical data from a Binance file based on the file URL. 

### Args

| Parameter    | Type        | Description                              |
|--------------|-------------|------------------------------------------|
| `file_url`   | `str`       | The URL of the Binance file.             |
| `has_header` | `bool`      | Whether the file has a header.           |
| `columns`    | `List[str]` | The columns to be included in the data.  |

### Returns

`self.data` (pl.DataFrame)

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

`self.data` (pl.DataFrame)