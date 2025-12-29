from clickhouse_connect import get_client
import polars as pl
import time
from typing import Optional, Tuple


def get_trades_data(month_year: Optional[Tuple[int,int]] = None,
                    n_latest: Optional[int] = None,
                    n_random: Optional[int] = None,
                    include_datetime_col: bool = True,
                    show_summary: bool = False) -> pl.DataFrame:
    
    '''
    Query Binance raw trades data from ClickHouse.
    
    Args:
        month_year (tuple[int,int] | None): (month, year) to fetch, e.g. (3, 2025)
        n_latest (int | None): if set, fetch this many latest rows instead
        n_random (int | None): if set, fetch this n_random of random samples
        include_datetime_col (bool): whether to include `datetime` in the result
        show_summary (bool): if a summary for data is printed out
        
    Returns:
        pl.DataFrame: the requested trades
    '''
    
    client = get_client(host='localhost',
                        port=8123,
                        username='default',
                        password='password123',
                        compression=True)

    select_cols = ['trade_id', 'timestamp', 'price', 'quantity', 'is_buyer_maker']
    
    if include_datetime_col:
        select_cols.append('datetime')

    if month_year and n_latest is None and n_random is None:
        month, year = month_year
        where = (f"WHERE datetime >= toDateTime('{year:04d}-{month:02d}-01 00:00:00') "
                 f"AND datetime <  addMonths(toDateTime('{year:04d}-{month:02d}-01 00:00:00'),1)")

    elif n_latest and n_random is None:
        where = f"ORDER BY toStartOfDay(datetime) DESC, trade_id DESC LIMIT {n_latest}"

    elif n_random:        
        where = f"ORDER BY sipHash64(tuple(trade_id, timestamp)) LIMIT {n_random}"
    
    else:
        raise AttributeError('Invalid parameter combination: Exactly one of `month_year`, `n_latest`, or `n_random` must be set.'
                             'Ensure that only one of these parameters is provided and the others are None.')
    query = (f"SELECT {', '.join(select_cols)} "
             f"FROM tdw.binance_trades {where}")

    start = time.time()
    arrow_table = client.query_arrow(query)
    polars_df = pl.from_arrow(arrow_table)
    polars_df = polars_df.sort('trade_id')
    
    polars_df = polars_df.with_columns([
        (pl.col('datetime').cast(pl.Int64) * 1000)
          .cast(pl.Datetime('ms', time_zone='UTC'))
          .alias('datetime')])

    elapsed = time.time() - start

    if show_summary is True:
        print(f"{elapsed:.2f} seconds | {polars_df.shape[0]} rows | {polars_df.shape[1]} columns | {polars_df.estimated_size()/(1024**3):.2f} GB RAM")

    return polars_df