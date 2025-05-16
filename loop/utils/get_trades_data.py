from clickhouse_connect import get_client
import polars as pl
import time
from typing import Optional, Tuple

def get_trades_data(month_year: Optional[Tuple[int,int]] = None,
                    n_rows: Optional[int] = None,
                    include_datetime_col: bool = True,
                    show_summary: bool = False) -> pl.DataFrame:
    
    '''Get Binance raw trades data.

    Args:
        month_year (tuple[int,int] | None): (month, year) to fetch, e.g. (3, 2025).
        n_rows (int | None): if set, fetch this many latest rows instead.
        include_datetime_col (bool): whether to include `datetime` in the result.
        show_summary (bool): if a summary for data is printed out

    Returns:
        pl.DataFrame: the requested trades.
    '''
    
    client = get_client(
        host='localhost',
        port=8123,
        username='default',
        password='password123',
        compression=True
    )

    select_cols = [
        'trade_id', 'timestamp', 'price', 'quantity', 'is_buyer_maker'
    ]
    if include_datetime_col:
        select_cols.append('datetime')

    if month_year is not None and n_rows is None:
        month, year = month_year
        where = (
            f"WHERE datetime >= toDateTime('{year:04d}-{month:02d}-01 00:00:00') "
            f"AND datetime <  addMonths(toDateTime('{year:04d}-{month:02d}-01 00:00:00'),1)"
        )
    elif n_rows is not None and month_year is None:
        where = f"ORDER BY toStartOfDay(datetime) DESC, trade_id DESC LIMIT {n_rows}"
    else:
        raise AttributeError('Either month_year or n_rows must be set, not both.')

    query = (
        f"SELECT {', '.join(select_cols)} "
        f"FROM tdw.binance_trades {where}"
    )

    start = time.time()
    arrow_table = client.query_arrow(query)
    polars_df = pl.from_arrow(arrow_table)
    polars_df = polars_df.sort("trade_id")
    
    polars_df = polars_df.with_columns([
        (pl.col('datetime').cast(pl.Int64) * 1000)
          .cast(pl.Datetime("ms", time_zone="UTC"))
          .alias("datetime")])

    elapsed = time.time() - start

    if show_summary is True:
        print(f"{elapsed:.2f} seconds | {polars_df.shape[0]} rows | {polars_df.shape[1]} columns | {polars_df.estimated_size()/(1024**3):.2f} GB RAM")

    return polars_df
