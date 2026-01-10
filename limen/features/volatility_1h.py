import polars as pl


def volatility_1h(df: pl.DataFrame, volatility_column: str = 'returns_volatility_12') -> pl.DataFrame:
    '''
    Compute volatility_1h alias from existing volatility column.
    
    Args:
        df (pl.DataFrame): Klines dataset with volatility column
        volatility_column (str): Name of the source volatility column
        
    Returns:
        pl.DataFrame: The input data with a new column 'volatility_1h'
    '''
    
    return df.with_columns([
        pl.col(volatility_column).alias('volatility_1h')
    ])