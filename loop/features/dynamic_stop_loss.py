import polars as pl


def dynamic_stop_loss(data: pl.DataFrame, 
                     base_stop_loss: float,
                     stop_volatility_multiplier: float) -> pl.DataFrame:
    
    '''
    Compute dynamic stop loss levels based on volatility conditions and regime.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'volatility_measure', 'regime_multiplier' columns
        base_stop_loss (float): Base stop loss threshold
        stop_volatility_multiplier (float): Multiplier for volatility-adjusted stops
        
    Returns:
        pl.DataFrame: The input data with a new column 'dynamic_stop_loss'
    '''
    
    return data.with_columns([
        (pl.col('volatility_measure') * stop_volatility_multiplier * pl.col('regime_multiplier'))
        .clip(base_stop_loss * 0.7, base_stop_loss * 1.4)
        .alias('dynamic_stop_loss')
    ])