import polars as pl
from loop.indicators.window_return import window_return


def window_return_regime(df: pl.DataFrame, period: int = 24, r_hi: float = 0.0, r_lo: float = 0.0) -> pl.DataFrame:

    '''
    Classify regime using windowed return close/close.shift(period)-1.

    Args:
        df (pl.DataFrame): Input with 'close'
        period (int): Return window
        r_hi (float): Upper threshold for Up
        r_lo (float): Lower threshold for Down

    Returns:
        pl.DataFrame: With 'regime_window_return' in {"Up","Flat","Down"}
    '''

    ret_col = f'ret_{period}'
    df2 = window_return(df, period)
    return df2.with_columns([
        pl.when(pl.col(ret_col) >= r_hi).then(pl.lit('Up'))
         .when(pl.col(ret_col) <= r_lo).then(pl.lit('Down'))
         .otherwise(pl.lit('Flat')).alias('regime_window_return')
    ])


