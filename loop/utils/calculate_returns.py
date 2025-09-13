import polars as pl


def calculate_returns_if_missing(df: pl.DataFrame) -> pl.DataFrame:
    
    '''
    Compute returns column if missing from the dataset.
    
    Args:
        df (pl.DataFrame): Klines dataset with 'close' column
        
    Returns:
        pl.DataFrame: The input data with a new column 'returns' (if not already present)
    '''
    if 'returns' not in df.columns:
        df = df.with_columns([
            (pl.col('close') / pl.col('close').shift(1) - 1).alias('returns')
        ])
    return df