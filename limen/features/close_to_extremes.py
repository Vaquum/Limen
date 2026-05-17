import polars as pl

EPSILON = 1e-10


def close_to_extremes(data: pl.DataFrame) -> pl.DataFrame:

    '''
    Compute close position relative to high and low extremes.

    Args:
        data (pl.DataFrame): Klines dataset with 'close', 'high', 'low' columns

    Returns:
        pl.DataFrame: The input data with new columns 'close_to_high' and 'close_to_low'
    '''

    return data.with_columns([
        ((pl.col('close') - pl.col('high')) / (pl.col('high') + EPSILON)).alias('close_to_high'),
        ((pl.col('close') - pl.col('low')) / (pl.col('low') + EPSILON)).alias('close_to_low')
    ])
