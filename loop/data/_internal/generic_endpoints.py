from clickhouse_connect import get_client
import polars as pl
import time
from typing import Optional, Tuple, Dict


def query_raw_data(table_name: str,
                   id_col: str,
                   select_cols: list,
                   month_year: Optional[Tuple[int, int]] = None,
                   n_rows: Optional[int] = None,
                   n_random: Optional[int] = None,
                   include_datetime_col: bool = True,
                   show_summary: bool = False) -> pl.DataFrame:
    '''
    Query ClickHouse table and return results as Polars DataFrame.

    Args:
        table_name (str): ClickHouse table name (e.g., 'binance_trades', 'binance_futures_trades')
        id_col (str): ID column for sorting (e.g., 'trade_id', 'futures_trade_id', 'agg_trade_id')
        select_cols (list): Columns to select (datetime added automatically if include_datetime_col=True)
        month_year (tuple, optional): (month, year) tuple for month-based filtering
        n_rows (int, optional): Number of latest rows to fetch
        n_random (int, optional): Number of random rows to fetch
        include_datetime_col (bool, optional): Whether to include the datetime column
        show_summary (bool, optional): Whether to print query summary

    Returns:
        pl.DataFrame: Polars DataFrame with the query results
    '''

    client = get_client(host='localhost',
                        port=8123,
                        username='default',
                        password='password123',
                        compression=True)

    cols = select_cols.copy()
    if include_datetime_col and 'datetime' not in cols:
        cols.append('datetime')

    param_count = sum([month_year is not None, n_rows is not None, n_random is not None])
    if param_count != 1:
        raise ValueError(
            f"Exactly one of month_year, n_rows, or n_random must be provided. "
            f"Got: month_year={month_year}, n_rows={n_rows}, n_random={n_random}"
        )

    if month_year is not None:
        month, year = month_year
        where = (
            f"WHERE datetime >= toDateTime('{year:04d}-{month:02d}-01 00:00:00') "
            f"AND datetime < addMonths(toDateTime('{year:04d}-{month:02d}-01 00:00:00'), 1)"
        )
    elif n_rows is not None:
        where = f"ORDER BY toStartOfDay(datetime) DESC, {id_col} DESC LIMIT {n_rows}"
    elif n_random is not None:
        where = f"ORDER BY sipHash64(tuple({id_col}, timestamp)) LIMIT {n_random}"

    query = f"SELECT {', '.join(cols)} FROM tdw.{table_name} {where}"

    start = time.time()
    arrow_table = client.query_arrow(query)
    polars_df = pl.from_arrow(arrow_table)
    polars_df = polars_df.sort(id_col)

    if include_datetime_col:
        polars_df = polars_df.with_columns([
            (pl.col('datetime').cast(pl.Int64) * 1000)
            .cast(pl.Datetime('ms', time_zone='UTC'))
            .alias('datetime')
        ])

    elapsed = time.time() - start

    if show_summary:
        print(f"{elapsed:.2f}s | {polars_df.shape[0]} rows | "
              f"{polars_df.shape[1]} cols | "
              f"{polars_df.estimated_size()/(1024**3):.2f} GB RAM")

    return polars_df


def query_klines_data(n_rows: Optional[int] = None,
                      kline_size: int = 1,
                      start_date_limit: str = None,
                      futures: bool = False,
                      show_summary: bool = False) -> pl.DataFrame:

    '''
    Query Binance klines data from ClickHouse.

    Args:
        n_rows (int | None): if not None, fetch this many latest rows instead
        kline_size (int): the size of the kline in seconds
        start_date_limit (str): the start date of the klines data
        futures (bool): if the data is from futures
        show_summary (bool): if a summary for data is printed out

    Returns:
        pl.DataFrame: the requested klines data
    '''

    client = get_client(
        host='localhost',
        port=8123,
        username='default',
        password='password123',
        compression=True
    )

    if n_rows is not None:
        limit = f'LIMIT {n_rows}'
    else:
        limit = ''

    if start_date_limit is not None:
        start_date_limit = f"WHERE datetime >= toDateTime('{start_date_limit}') "
    else:
        start_date_limit = ''

    if futures is True:
        db_table = 'FROM tdw.binance_futures_trades '
        id_col = 'futures_trade_id'
    else:
        db_table = 'FROM tdw.binance_trades '
        id_col = 'trade_id'

    query = (
        f'SELECT '
        f'    toDateTime({kline_size} * intDiv(toUnixTimestamp(datetime), {kline_size})) AS datetime, '
        f'    argMin(price, {id_col})       AS open, '
        f'    max(price)                    AS high, '
        f'    min(price)                    AS low, '
        f'    argMax(price, {id_col})       AS close, '
        f'    avg(price)                    AS mean, '
        f'    stddevPopStable(price)        AS std, '
        f'    quantileExact(0.5)(price)     AS median, '
        f'    quantileExact(0.75)(price) - quantileExact(0.25)(price) AS iqr, '
        f'    sumKahan(quantity)            AS volume, '
        f'    avg(is_buyer_maker)           AS maker_ratio, '
        f'    count()                       AS no_of_trades, '
        f'    argMin(price * quantity, {id_col})    AS open_liquidity, '
        f'    max(price * quantity)         AS high_liquidity, '
        f'    min(price * quantity)         AS low_liquidity, '
        f'    argMax(price * quantity, {id_col})    AS close_liquidity, '
        f'    sum(price * quantity)         AS liquidity_sum, '
        f'    sumKahan(is_buyer_maker * quantity)   AS maker_volume, '
        f'    sum(is_buyer_maker * price * quantity) AS maker_liquidity '
        f'{db_table}'
        f'{start_date_limit}'
        f'GROUP BY datetime '
        f'ORDER BY datetime ASC '
        f'{limit}'
    )

    start = time.time()
    arrow_table = client.query_arrow(query)
    polars_df = pl.from_arrow(arrow_table)

    polars_df = polars_df.with_columns([
        (pl.col('datetime').cast(pl.Int64) * 1000)
          .cast(pl.Datetime('ms', time_zone='UTC'))
          .alias('datetime')])

    polars_df = polars_df.with_columns([
        pl.col('mean').round(5),
        pl.col('std').round(6),
        pl.col('volume').round(9),
        pl.col('liquidity_sum').round(1),
        pl.col('maker_liquidity').round(1),
    ])

    polars_df = polars_df.sort('datetime')

    elapsed = time.time() - start

    if show_summary is True:
        print(f'{elapsed:.2f} seconds | {polars_df.shape[0]} rows | {polars_df.shape[1]} columns | {polars_df.estimated_size()/(1024**3):.2f} GB RAM')

    return polars_df
