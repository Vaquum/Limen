import polars as pl

CMP_N_100000 = 100000
CMP_N_2 = 2

WILLR_SCALE = -100.0


def willr(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    period: int = 14,
) -> pl.DataFrame:

    '''
    Compute Williams' %R (WILLR).

    Args:
        data (pl.DataFrame): Dataset with high/low/close columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices
        period (int): Number of periods (2..100000)

    Returns:
        pl.DataFrame: The input data with a new column 'willr_{period}'
    '''

    if period < CMP_N_2 or period > CMP_N_100000:
        raise ValueError('willr period must be between 2 and 100000')

    out_col = f"willr_{period}"
    highest = pl.col(high_col).rolling_max(window_size=period)
    lowest = pl.col(low_col).rolling_min(window_size=period)
    diff = (highest - lowest) / WILLR_SCALE

    willr_expr = (
        pl.when(pl.int_range(0, pl.len()) < (period - 1))
        .then(None)
        .when(diff != 0.0)
        .then((highest - pl.col(close_col)) / diff)
        .otherwise(0.0)
        .alias(out_col)
    )

    return data.with_columns(willr_expr)
