import polars as pl


def close_position(data: pl.DataFrame) -> pl.DataFrame:
    
    '''
    Compute close position within the high-low range as percentage.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'high', 'low', and 'close' columns

    Returns:
        pl.DataFrame: The input data with a new column 'close_position'
    '''

    return data.with_columns(((pl.col('close') - pl.col('low')) / (pl.col('high') - pl.col('low') + 1e-8)).alias('close_position'))