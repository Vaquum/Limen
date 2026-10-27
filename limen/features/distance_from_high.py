import polars as pl
import limen._feature_constraints as _check


def distance_from_high(data: pl.DataFrame, period: int = 24) -> pl.DataFrame:

    '''
    Compute distance from rolling high as percentage.

    Args:
        data (pl.DataFrame): Klines dataset with 'high', 'close' columns
        period (int): Number of periods for rolling high calculation

    Returns:
        pl.DataFrame: The input data with a new column 'distance_from_high'
    '''

    _check.integers(lower=1, period=period)

    return data.with_columns([
        ((pl.col('high').rolling_max(window_size=period) - pl.col('close')) / pl.col('close'))
        .alias('distance_from_high')
    ])
