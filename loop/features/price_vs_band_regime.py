import polars as pl


def price_vs_band_regime(df: pl.DataFrame, period: int = 24, band: str = 'std', k: float = 0.75) -> pl.DataFrame:

    '''
    Classify regime by comparing close to center +/- k*band_width.
    Band width can be rolling std(period) or deviation std around SMA(period).

    Args:
        df (pl.DataFrame): Input with 'close' (and optionally 'mean','std','iqr')
        period (int): Rolling period
        band (str): 'std' or 'dev_std'
        k (float): Band multiplier

    Returns:
        pl.DataFrame: With 'regime_price_band' in {"Up","Flat","Down"}
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


