import polars as pl


def ma_slope_regime(df: pl.DataFrame,
                    period: int = 24,
                    threshold: float = 0.0,
                    normalize_by_std: bool = True) -> pl.DataFrame:

    '''
    Compute regime using the slope of SMA(close, period) with optional normalization.

    Args:
        df (pl.DataFrame): Klines dataset with 'close' column
        period (int): SMA period for the slope
        threshold (float): Slope threshold; applied after normalization when enabled
        normalize_by_std (bool): Whether to divide slope by rolling std(period)

    Returns:
        pl.DataFrame: The input data with a new column 'regime_ma_slope'
    '''

    sma = pl.col('close').rolling_mean(window_size=period)
    slope = (sma - sma.shift(period)) / period
    
    if normalize_by_std:
        slope = slope / (pl.col('close').rolling_std(window_size=period) + 1e-12)

    return df.with_columns([
        pl.when(slope > threshold).then(pl.lit('Up'))
         .when(slope < -threshold).then(pl.lit('Down'))
         .otherwise(pl.lit('Flat')).alias('regime_ma_slope')
    ])
