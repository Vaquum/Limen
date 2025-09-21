import polars as pl


def spread_percent(data: pl.DataFrame) -> pl.DataFrame:
    
    '''
    Compute high-low spread as percentage of close price.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'high', 'low', 'close' columns
        
    Returns:
        pl.DataFrame: The input data with a new column 'spread_percent'
    '''
    
    return data.with_columns([
        ((pl.col('high') - pl.col('low')) / pl.col('close'))
        .alias('spread_percent')
    ])