from clickhouse_connect import get_client
import polars as pl
import time
from typing import Optional

def get_klines_data(n_rows: Optional[int] = None,
                    kline_size: int = 1,
                    start_date_limit: str = None,
                    futures: bool = False,
                    show_summary: bool = False,) -> pl.DataFrame:
    
    '''Get 1 second klines data based on Binance raw trades data. Returns either 
       everything or n_rows. Everything is 

    Args:
        n_rows (int | None): if not None, fetch this many latest rows instead.
        kline_size (int): the size of the kline in seconds.
        start_date_limit (str): the start date of the klines data.
        futures (bool): if the data is from futures.
        show_summary (bool): if a summary for data is printed out.

    Returns:
        pl.DataFrame: the requested klines data.
    '''
    
    client = get_client(
        host='localhost',
        port=8123,
        username='default',
        password='password123',
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
        db_table = f"FROM tdw.binance_futures_trades "
    else:
        db_table = f"FROM tdw.binance_trades "

    query = (
        f"SELECT "
        f"    toDateTime(toStartOfMinute(datetime) + {kline_size} * intDiv(toSecond(datetime), {kline_size})) AS datetime, "
        f"    first_value(price)            AS open, "
        f"    max(price)                    AS high, "
        f"    min(price)                    AS low, "
        f"    last_value(price)             AS close, "
        f"    sum(quantity)                 AS volume, "
        f"    avg(is_buyer_maker)           AS maker_ratio, "
        f"    count()                       AS no_of_trades, "
        f"    first_value(price * quantity) AS open_liquidity, "
        f"    max(price * quantity)         AS high_liquidity, "
        f"    min(price * quantity)         AS low_liquidity, "
        f"    last_value(price * quantity)  AS close_liquidity, "
        f"    sum(price * quantity)         AS liquidity_sum "
        f"{db_table}"
        f"{start_date_limit}"
        f"GROUP BY datetime "
        f"ORDER BY datetime ASC {limit}"
    )

    start = time.time()
    arrow_table = client.query_arrow(query)
    polars_df = pl.from_arrow(arrow_table)
    
    polars_df = polars_df.with_columns([
        (pl.col('datetime').cast(pl.Int64) * 1000)
          .cast(pl.Datetime("ms", time_zone="UTC"))
          .alias("datetime")])

    polars_df = polars_df.sort("datetime")

    elapsed = time.time() - start

    if show_summary is True:
        print(f"{elapsed:.2f} seconds | {polars_df.shape[0]} rows | {polars_df.shape[1]} columns | {polars_df.estimated_size()/(1024**3):.2f} GB RAM")

    return polars_df
