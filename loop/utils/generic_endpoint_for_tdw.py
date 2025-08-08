from clickhouse_connect import get_client
import polars as pl
import time
from typing import Optional, Tuple


def generic_endpoint_for_tdw(month_year: Optional[Tuple[int,int]] = None,
                             n_rows: Optional[int] = None,
                             include_datetime_col: bool = True,
                             select_cols: list = None,
                             table_name: str = None,
                             sort_by: str = None,
                             show_summary: bool = False) -> pl.DataFrame:

    '''
    Query ClickHouse table and return results as Polars DataFrame.
    
    Args:
        month_year (tuple, optional): Month and year to filter by (year, month)
        n_rows (int, optional): Number of rows to return
        include_datetime_col (bool, optional): Whether to include the datetime column
        select_cols (list, optional): Columns to select
        table_name (str, optional): Name of the table to query
        sort_by (str, optional): Column to sort by
        show_summary (bool, optional): Whether to show a summary of the query
        
    Returns:
        pl.DataFrame: Polars DataFrame with the query results
    '''

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
        raise ValueError('Exactly one of month_year or n_rows must be provided, not both or neither.')

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
