import polars as pl


def standardize_datetime_column(df: pl.DataFrame) -> pl.DataFrame:
    
    '''
    Compute standardized datetime column format across datasets.
    
    Args:
        df (pl.DataFrame): Dataset with 'Date' or 'datetime' column
        
    Returns:
        pl.DataFrame: The input data with standardized 'datetime' column of Datetime type
    '''
    if 'Date' in df.columns:
        df = df.rename({'Date': 'datetime'})
    
    if df.schema['datetime'] != pl.Datetime:
        df = df.with_columns(pl.col('datetime').str.to_datetime())
    
    return df