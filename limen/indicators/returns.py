from limen._parallel_contract import attach as _parallel
import polars as pl


def returns(data: pl.DataFrame) -> pl.DataFrame:

    '''
    Compute period-over-period returns of close prices.

    Returns:
        pl.DataFrame: The input data with a new column 'returns'
    '''

    return data.with_columns(pl.col('close').pct_change().alias('returns'))

_parallel(returns, backend_imports=('polars',))
