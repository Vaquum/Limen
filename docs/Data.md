# Data

A crucial key point here is that all individual contirbutors work based on the same underlying data. We achieve this by always calling data from the provided endpoints. If you don't find what you need through these endpoints, [make an issue](https://github.com/Vaquum/Loop/issues/new) that requests the data that you need. 

All of the endpoints are found in [`loop/data.py`](../loop/data.py). 

All of the endpoints rely on [Binance Market Data](https://data.binance.vision/?prefix=) as their source.

There are in total five distinct data endpoints:

- `HistoricalData.get_binance_file`
- `HistoricalData.get_historical_klines`
- `HistoricalData.get_historical_trades`
- `HistoricalData.get_historical_agg_trades`
- `HistoricalData.get_historical_futures_trades`

All of these endpoints are available through: 

```
# Initiatilize the API
import loop
historical = loop.HistoricalData()

# Call one of the endpoints:
historical.get_historical_klines()

# Access the data
historical.data
```

All of the endpoints return `pl.DataFrame`. 

NOTE: There is an important performance cost that one suffers from moving the data out of the `pl.DataFrame` (e.g. to `pd.DataFrame`). 

## `HistoricalData.get_binance_file`

Get data from a Binance file based on the file URL.

### Args

`file_url` (str): The URL of the Binance file
`has_header` (bool): Whether the file has a header
`columns` (List[str]): The columns to be included in the data

### Returns

`self.data` (pl.DataFrame)

## `HistoricalData.get_historical_klines`

Get historical klines data from Binance trades data.

### Args

`n_rows` (int): Number of rows to be pulled
`kline_size` (int): Size of the kline in seconds
`start_date_limit` (str): The start date of the klines data
`futures` (bool): if the data is from futures.

### Returns

`self.data` (pl.DataFrame)

## `HistoricalData.get_historical_trades`

Get historical trades data from `tdw.binance_trades` table in Clickhouse.

### Args

`month_year` (Tuple): The month of data to be pulled e.g. (3, 2025)
`n_latest` (int): Number of latest rows to be pulled
`n_random` (int): Number of random rows to be pulled
`include_datetime_col` (bool): If datetime column is to be included

### Returns

`self.data` (pl.DataFrame)

## `HistoricalData.get_historical_agg_trades`

Get historical aggTrades data from `tdw.binance_agg_trades` table in Clickhouse.

### Args

`month_year` (Tuple): The month of data to be pulled e.g. (3, 2025)
`n_rows` (int): Number of rows to be pulled
`include_datetime_col` (bool): If datetime column is to be included

### Returns

`self.data` (pl.DataFrame)

## `HistoricalData.get_historical_futures_trades`

Get Binance futures trades data from `tdw.binance_futures_trades` table in Clickhouse.

### Args

`month_year` (tuple[int,int] | None): (month, year) to fetch, e.g. (3, 2025).
`n_rows` (int | None): if set, fetch this many latest rows instead.
`include_datetime_col` (bool): whether to include `datetime` in the result.
`show_summary` (bool): if a summary for data is printed out

### Returns

`self.data` (pl.DataFrame)