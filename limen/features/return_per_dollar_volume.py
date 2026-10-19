from limen._parallel_contract import attach as _parallel
import polars as pl

EPSILON = 1e-10


def return_per_dollar_volume(
    data: pl.DataFrame,
    price_col: str = 'close',
    volume_col: str = 'volume',
) -> pl.DataFrame:

    '''
    Compute signed return per traded dollar as a simple impact-efficiency proxy.

    Args:
        volume_col (str): Column name for traded volume

    Returns:
        pl.DataFrame: The input data with a new column 'return_per_dollar_volume'
    '''

    dollar_volume_expr = pl.col(price_col) * pl.col(volume_col)

    return data.with_columns(
        (pl.col(price_col).pct_change() / (dollar_volume_expr + EPSILON)).alias('return_per_dollar_volume')
    )

_parallel(return_per_dollar_volume, backend_imports=('polars',))
