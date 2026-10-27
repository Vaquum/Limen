import polars as pl
import limen._feature_constraints as _check


def wclprice(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Weighted Close Price.

    Args:
        data (pl.DataFrame): Dataset with high, low, and close columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'wclprice'
    '''

    _check.names(high_col=high_col, low_col=low_col, close_col=close_col)

    wcl_price = (
        pl.col(high_col).cast(pl.Float64)
        + pl.col(low_col).cast(pl.Float64)
        + (pl.col(close_col).cast(pl.Float64) * 2.0)
    ) / 4.0

    return data.with_columns(wcl_price.alias('wclprice'))
