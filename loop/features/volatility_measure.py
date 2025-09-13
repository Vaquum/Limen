import polars as pl


def volatility_measure(data: pl.DataFrame) -> pl.DataFrame:
    
    '''
    Compute combined volatility measure from rolling volatility and ATR percentage.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'rolling_volatility', 'atr_percent_sma' columns
        
    Returns:
        pl.DataFrame: The input data with a new column 'volatility_measure'
    '''
    
    return data.with_columns([
        ((pl.col('rolling_volatility') + pl.col('atr_percent_sma')) / 2)
        .alias('volatility_measure')
    ])