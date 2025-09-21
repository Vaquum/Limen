import polars as pl


def time_decay(df: pl.DataFrame, 
               time_column: str, 
               halflife: float,
               time_units: float = 1.0,
               default_value: float = 0.5,
               output_column: str = 'time_decay_factor') -> pl.DataFrame:
    
    '''
    Compute exponential time decay factor based on time elapsed.
    
    Args:
        df (pl.DataFrame): Dataset with time column
        time_column (str): Column name containing time values
        halflife (float): Half-life for exponential decay
        time_units (float): Conversion factor for time units
        default_value (float): Default value when time column is null
        output_column (str): Name for output decay factor column
        
    Returns:
        pl.DataFrame: The input data with new time decay factor column
    '''
    
    adjusted_halflife = halflife / time_units
    
    df = df.with_columns([
        pl.when(pl.col(time_column).is_not_null())
            .then((pl.lit(2).log() * pl.lit(-1)).mul(pl.col(time_column)).truediv(adjusted_halflife).exp())
            .otherwise(pl.lit(default_value))
            .alias(output_column)
    ])
    
    return df