import polars as pl


def ma_slope_regime(df: pl.DataFrame, period: int = 24, threshold: float = 0.0, normalize_by_std: bool = True) -> pl.DataFrame:

    '''
    Classify regime by slope of SMA(close, period). Optionally normalized by rolling std.

    Args:
        df (pl.DataFrame): Input with 'close'
        period (int): SMA period
        threshold (float): Slope threshold; if normalize_by_std, applies to normalized slope
        normalize_by_std (bool): Normalize slope by rolling std(period)

    Returns:
        pl.DataFrame: With column 'regime_ma_slope' in {"Up","Flat","Down"}
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


