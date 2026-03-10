import polars as pl

TRANGE_LOOKBACK = 1


def trange(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute True Range (TRANGE).

    Args:
        data (pl.DataFrame): Klines dataset with high/low/close columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'trange'
    '''

    prev_close = pl.col(close_col).shift(1)
    true_range = (
        pl.max_horizontal(pl.col(high_col), prev_close)
        - pl.min_horizontal(pl.col(low_col), prev_close)
    )

    return data.with_columns(
        pl.when(pl.int_range(0, pl.len()) < TRANGE_LOOKBACK)
        .then(None)
        .otherwise(true_range)
        .alias('trange')
    )
