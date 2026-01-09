import polars as pl


def gap_high(data: pl.DataFrame) -> pl.DataFrame:
    
    '''
    Compute gap between current high and previous close as percentage.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'high' and 'close' columns

    Returns:
        pl.DataFrame: The input data with a new column 'gap_high'
    '''

    return data.with_columns(((pl.col('high') - pl.col('close').shift(1)) / pl.col('close').shift(1)).alias('gap_high'))