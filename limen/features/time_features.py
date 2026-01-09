import polars as pl


def time_features(df: pl.DataFrame) -> pl.DataFrame:

    '''
    Compute hour and minute features from datetime column.
    
    Args:
        df (pl.DataFrame): Klines dataset with 'datetime' column
        
    Returns:
        pl.DataFrame: The input data with new columns 'hour', 'minute', 'weekday'
    '''
    
    return df.with_columns([
        pl.col('datetime').dt.hour().alias('hour'),
        pl.col('datetime').dt.minute().alias('minute'),
        pl.col('datetime').dt.weekday().alias('weekday')
    ])