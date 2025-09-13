import polars as pl
from typing import Literal


def price_vs_band_regime(df: pl.DataFrame,
                         period: int = 24,
                         band: Literal['std', 'dev_std'] = 'std',
                         k: float = 0.75) -> pl.DataFrame:

    '''
    Compute regime by comparing 'close' to center ± k × band width over a rolling window.

    Args:
        df (pl.DataFrame): Klines dataset with 'close' column
        period (int): Rolling period for center and band width
        band (Literal['std', 'dev_std']): Band width type to use
        k (float): Band multiplier applied to the width

    Returns:
        pl.DataFrame: The input data with a new column 'regime_price_band'
    '''

    center = pl.col('close').rolling_mean(window_size=period)
    
    if band == 'dev_std':
        dev = pl.col('close') - center
        width = dev.rolling_std(window_size=period)
    else:
        width = pl.col('close').rolling_std(window_size=period)

    upper = center + k * width
    lower = center - k * width

    return df.with_columns([
        pl.when(pl.col('close') > upper).then(pl.lit('Up'))
         .when(pl.col('close') < lower).then(pl.lit('Down'))
         .otherwise(pl.lit('Flat')).alias('regime_price_band')
    ])
