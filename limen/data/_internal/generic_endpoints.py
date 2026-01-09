from clickhouse_connect import get_client
import polars as pl
import time
from typing import Optional, Tuple


def query_raw_data(table_name: str,
                   id_col: str,
                   select_cols: list,
                   month_year: Optional[Tuple[int, int]] = None,
                   n_rows: Optional[int] = None,
                   n_random: Optional[int] = None,
                   include_datetime_col: bool = True,
                   show_summary: bool = False,
                   auth_token: str = None) -> pl.DataFrame:

    '''
    Query raw trade data from ClickHouse database.

    Args:
        table_name (str): ClickHouse table name
        id_col (str): ID column name for database table
        select_cols (list): Column names to select from table
        month_year (Optional[Tuple[int, int]]): Month and year tuple for filtering
        n_rows (Optional[int]): Number of latest rows to fetch
        n_random (Optional[int]): Number of random rows to fetch
        include_datetime_col (bool): Include datetime column in result
        show_summary (bool): Print query execution summary
        auth_token (str): Authentication token for ClickHouse database
    Returns:
        pl.DataFrame: Raw trade data with selected columns sorted by ID
    '''

    client = get_client(host='localhost',
                        port=8123,
                        username='default',
                        password=auth_token,
                        compression=True)

    cols = select_cols.copy()

    param_count = sum([month_year is not None, n_rows is not None, n_random is not None])
    if param_count != 1:
        raise ValueError(
            f"Exactly one of month_year, n_rows, or n_random must be provided. "
            f"Got: month_year={month_year}, n_rows={n_rows}, n_random={n_random}"
        )

    datetime_requested = include_datetime_col or 'datetime' in select_cols
    datetime_needed_for_query = month_year is not None or n_rows is not None

    timestamp_requested = 'timestamp' in select_cols
    timestamp_needed_for_query = n_random is not None

    if datetime_needed_for_query and 'datetime' not in cols:
        cols.append('datetime')

    if timestamp_needed_for_query and 'timestamp' not in cols:
        cols.append('timestamp')

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

    if 'timestamp' in polars_df.columns:
        polars_df = polars_df.with_columns([
            pl.when(pl.col('timestamp') < 10**13)
            .then(pl.col('timestamp'))
            .otherwise(pl.col('timestamp') // 1000)
            .cast(pl.UInt64)
            .alias('timestamp')
        ])

    if 'datetime' in polars_df.columns:
        polars_df = polars_df.with_columns([
            (pl.col('datetime').cast(pl.Int64) * 1000)
            .cast(pl.Datetime('ms', time_zone='UTC'))
            .alias('datetime')
        ])

    elapsed = time.time() - start

    if show_summary:
        print(f"{elapsed:.2f} s | {polars_df.shape[0]} rows | "
              f"{polars_df.shape[1]} cols | "
              f"{polars_df.estimated_size()/(1024**3):.2f} GB RAM")

    if not datetime_requested and 'datetime' in polars_df.columns:
        polars_df = polars_df.drop('datetime')

    if not timestamp_requested and 'timestamp' in polars_df.columns:
        polars_df = polars_df.drop('timestamp')

    return polars_df


def query_klines_data(n_rows: Optional[int] = None,
                      kline_size: int = 1,
                      start_date_limit: Optional[str] = None,
                      futures: bool = False,
                      show_summary: bool = False,
                      auth_token: str = None) -> pl.DataFrame:

    '''
    Query aggregated klines data from ClickHouse database.

    Args:
        n_rows (Optional[int]): Number of latest kline rows to fetch
        kline_size (int): Kline period size in seconds
        start_date_limit (Optional[str]): Start date for filtering klines data
        futures (bool): Query futures market data instead of spot
        show_summary (bool): Print query execution summary
        auth_token (str): Authentication token for ClickHouse database
    Returns:
        pl.DataFrame: Klines data with 19 columns including OHLC, volume, and liquidity metrics
    '''

    client = get_client(
        host='localhost',
        port=8123,
        username='default',
        password=auth_token,
        compression=True
    )

    if n_rows is not None:
        limit = f"LIMIT {n_rows}"
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
        f"SELECT "
        f"    toDateTime({kline_size} * intDiv(toUnixTimestamp(datetime), {kline_size})) AS datetime, "
        f"    argMin(price, {id_col})       AS open, "
        f"    max(price)                    AS high, "
        f"    min(price)                    AS low, "
        f"    argMax(price, {id_col})       AS close, "
        f"    avg(price)                    AS mean, "
        f"    stddevPopStable(price)        AS std, "
        f"    quantileExact(0.5)(price)     AS median, "
        f"    quantileExact(0.75)(price) - quantileExact(0.25)(price) AS iqr, "
        f"    sumKahan(quantity)            AS volume, "
        f"    avg(is_buyer_maker)           AS maker_ratio, "
        f"    count()                       AS no_of_trades, "
        f"    argMin(price * quantity, {id_col})    AS open_liquidity, "
        f"    max(price * quantity)         AS high_liquidity, "
        f"    min(price * quantity)         AS low_liquidity, "
        f"    argMax(price * quantity, {id_col})    AS close_liquidity, "
        f"    sum(price * quantity)         AS liquidity_sum, "
        f"    sumKahan(is_buyer_maker * quantity)   AS maker_volume, "
        f"    sum(is_buyer_maker * price * quantity) AS maker_liquidity "
        f"{db_table}"
        f"{start_date_limit}"
        f"GROUP BY datetime "
        f"ORDER BY datetime ASC "
        f"{limit}"
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

    if show_summary:
        print(f"{elapsed:.2f} s | {polars_df.shape[0]} rows | {polars_df.shape[1]} cols | {polars_df.estimated_size()/(1024**3):.2f} GB RAM")

    return polars_df
