import polars as pl
from limen.indicators.sma import sma
import limen._feature_constraints as _check

EPSILON = 1e-10


def volume_ratio(data: pl.DataFrame, period: int = 20) -> pl.DataFrame:

    '''
    Compute volume ratio relative to simple moving average baseline.

    Args:
        data (pl.DataFrame): Klines dataset with 'volume' column
        period (int): Number of periods for SMA calculation

    Returns:
        pl.DataFrame: The input data with a new column 'volume_ratio'
    '''

    # Calculate volume SMA if not already present
    _check.integers(lower=1, period=period)

    volume_sma_col = f'volume_sma_{period}'
    if volume_sma_col not in data.collect_schema().names():
        data = sma(data, 'volume', period)

    return data.with_columns([
        (pl.col('volume') / (pl.col(volume_sma_col) + EPSILON))
        .alias('volume_ratio')
    ])
