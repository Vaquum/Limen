import polars as pl

EPSILON = 1e-10


def wick_imbalance(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute upper-versus-lower wick imbalance as a fraction of total bar range.

    Args:
        data (pl.DataFrame): Klines dataset with open, high, low, and close columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'wick_imbalance'
    '''

    upper_wick = pl.col(high_col) - pl.max_horizontal(pl.col(open_col), pl.col(close_col))
    lower_wick = pl.min_horizontal(pl.col(open_col), pl.col(close_col)) - pl.col(low_col)

    return data.with_columns(
        ((upper_wick - lower_wick) / ((pl.col(high_col) - pl.col(low_col)) + EPSILON)).alias('wick_imbalance')
    )
