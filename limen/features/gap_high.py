from limen._parallel_contract import attach as _parallel
import polars as pl


def gap_high(data: pl.DataFrame) -> pl.DataFrame:

    '''
    Compute gap between current high and previous close as percentage.

    Returns:
        pl.DataFrame: The input data with a new column 'gap_high'
    '''

    return data.with_columns(((pl.col('high') - pl.col('close').shift(1)) / pl.col('close').shift(1)).alias('gap_high'))

_parallel(gap_high, backend_imports=('polars',))
