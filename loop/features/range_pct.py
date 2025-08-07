import polars as pl


def range_pct(data: pl.DataFrame) -> pl.DataFrame:
    
    '''
    Compute range as percentage of close price (high-low)/close.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'high', 'low', and 'close' columns

    Returns:
        pl.DataFrame: The input data with a new column 'range_pct'
    '''

    return data.with_columns(((pl.col('high') - pl.col('low')) / pl.col('close')).alias('range_pct'))