from clickhouse_connect import get_client
import polars as pl
import time
from typing import Optional, Tuple


def generic_endpoint_for_tdw(month_year: Optional[Tuple[int,int]] = None,
                             n_rows: Optional[int] = None,
                             include_datetime_col: bool = True,
                             show_summary: bool = False,
                             select_cols: list = None,
                             table_name: str = None,
                             sort_by: str = None) -> pl.DataFrame:

    '''For using as the base functionality for all tdw endpoints.'''

    client = get_client(host='localhost',
                        port=8123,
                        username='default',
                        password='password123',
                        compression=True)

    if include_datetime_col:
        select_cols.append('datetime')

    if month_year is not None and n_rows is None:
        month, year = month_year
        where = (
            f"WHERE datetime >= toDateTime('{year:04d}-{month:02d}-01 00:00:00') "
            f"AND datetime <  addMonths(toDateTime('{year:04d}-{month:02d}-01 00:00:00'),1)"
        )
    
    elif n_rows is not None and month_year is None:
        where = f"ORDER BY toStartOfDay(datetime) DESC, futures_trade_id DESC LIMIT {n_rows}"
    
    else:
        raise AttributeError('Either month_year or n_rows must be set, not both.')

    query = (
        f"SELECT {', '.join(select_cols)} "
        f"FROM tdw.{table_name} {where}"
    )

    start = time.time()
    arrow_table = client.query_arrow(query)
    polars_df = pl.from_arrow(arrow_table)
    polars_df = polars_df.sort(sort_by)
    
    if include_datetime_col:
        polars_df = polars_df.with_columns([
            (pl.col('datetime').cast(pl.Int64) * 1000)
            .cast(pl.Datetime("ms", time_zone="UTC"))
            .alias("datetime")])

    elapsed = time.time() - start

    if show_summary is True:
        print(f"{elapsed:.2f} seconds | {polars_df.shape[0]} rows | {polars_df.shape[1]} columns | {polars_df.estimated_size()/(1024**3):.2f} GB RAM")

    polars_df = polars_df.with_columns([
        pl.when(pl.col("timestamp") < 10**13)
        .then(pl.col("timestamp"))
        .otherwise(pl.col("timestamp") // 1000)
        .cast(pl.UInt64) 
        .alias("timestamp")
    ])

    return polars_df
