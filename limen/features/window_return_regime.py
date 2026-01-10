import polars as pl
from limen.indicators.window_return import window_return


def window_return_regime(df: pl.DataFrame,
                         period: int = 24,
                         r_hi: float = 0.0,
                         r_lo: float = 0.0) -> pl.DataFrame:

    '''
    Compute regime using windowed return close/close.shift(period) - 1.

    Args:
        df (pl.DataFrame): Klines dataset with 'close' column
        period (int): Window length for return calculation
        r_hi (float): Upper threshold for Up regime
        r_lo (float): Lower threshold for Down regime

    Returns:
        pl.DataFrame: The input data with a new column 'regime_window_return'
    '''

    ret_col = f'ret_{period}'
    df2 = window_return(df, period)
    
    return df2.with_columns([
        pl.when(pl.col(ret_col) >= r_hi).then(pl.lit('Up'))
         .when(pl.col(ret_col) <= r_lo).then(pl.lit('Down'))
         .otherwise(pl.lit('Flat')).alias('regime_window_return')
    ])
