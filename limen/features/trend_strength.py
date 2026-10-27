import polars as pl
import limen._feature_constraints as _check

EPSILON = 1e-10


def trend_strength(data: pl.DataFrame, fast_period: int = 20, slow_period: int = 50) -> pl.DataFrame:

    '''
    Compute trend strength based on moving average divergence.

    Args:
        data (pl.DataFrame): Klines dataset with 'close' column
        fast_period (int): Number of periods for fast SMA calculation
        slow_period (int): Number of periods for slow SMA calculation

    Returns:
        pl.DataFrame: The input data with a new column 'trend_strength'
    '''

    _check.integers(lower=1, fast_period=fast_period, slow_period=slow_period)

    return (
        data
        .with_columns([
            pl.col('close').rolling_mean(window_size=fast_period).alias('sma_fast'),
            pl.col('close').rolling_mean(window_size=slow_period).alias('sma_slow'),
        ])
        .with_columns([
            ((pl.col('sma_fast') - pl.col('sma_slow')) / (pl.col('sma_slow') + EPSILON)).alias('trend_strength')
        ])
        .drop(['sma_fast', 'sma_slow'])
    )
