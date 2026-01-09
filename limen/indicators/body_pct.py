import polars as pl


def body_pct(data: pl.DataFrame) -> pl.DataFrame:
    
    '''
    Compute the body percentage (candle body size relative to open).
    
    Args:
        data (pl.DataFrame): Klines dataset with 'open' and 'close' columns

    Returns:
        pl.DataFrame: The input data with a new column 'body_pct'
    '''

    return data.with_columns(((pl.col('close') - pl.col('open')) / pl.col('open')).alias('body_pct'))