import polars as pl

EPSILON = 1e-10


def body_to_range(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute candle body size relative to the full bar range.

    Args:
        data (pl.DataFrame): Klines dataset with open, high, low, and close columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'body_to_range'
    '''

    return data.with_columns(
        ((pl.col(close_col) - pl.col(open_col)).abs() / ((pl.col(high_col) - pl.col(low_col)) + EPSILON))
        .alias('body_to_range')
    )
