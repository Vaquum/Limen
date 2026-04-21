import polars as pl

EPSILON = 1e-10


def rejection_intensity(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute the wick fraction that rejects movement against the candle close direction.

    Args:
        data (pl.DataFrame): Klines dataset with open, high, low, and close columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'rejection_intensity'
    '''

    upper_wick = pl.col(high_col) - pl.max_horizontal(pl.col(open_col), pl.col(close_col))
    lower_wick = pl.min_horizontal(pl.col(open_col), pl.col(close_col)) - pl.col(low_col)
    bar_range = pl.col(high_col) - pl.col(low_col)

    rejection_component = (
        pl.when(pl.col(close_col) > pl.col(open_col))
        .then(lower_wick)
        .when(pl.col(close_col) < pl.col(open_col))
        .then(upper_wick)
        .otherwise(pl.max_horizontal(upper_wick, lower_wick))
    )

    return data.with_columns((rejection_component / (bar_range + EPSILON)).alias('rejection_intensity'))
