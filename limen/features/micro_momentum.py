import polars as pl
import limen._feature_constraints as _check


def micro_momentum(data: pl.DataFrame, period: int = 3) -> pl.DataFrame:

    '''
    Compute short-term price momentum over specified periods.

    Args:
        data (pl.DataFrame): Klines dataset with 'close' column
        period (int): Number of periods for momentum calculation

    Returns:
        pl.DataFrame: The input data with a new column 'micro_momentum'
    '''

    _check.integers(lower=0, period=period)

    return data.with_columns([
        pl.col('close').pct_change(period).alias('micro_momentum')
    ])
