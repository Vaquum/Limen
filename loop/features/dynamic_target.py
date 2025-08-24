import polars as pl


def dynamic_target(data: pl.DataFrame, 
                  base_min_breakout: float,
                  target_volatility_multiplier: float) -> pl.DataFrame:
    
    '''
    Compute dynamic target levels based on volatility conditions and regime.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'volatility_measure', 'regime_multiplier' columns
        base_min_breakout (float): Base minimum breakout threshold
        target_volatility_multiplier (float): Multiplier for volatility-adjusted targets
        
    Returns:
        pl.DataFrame: The input data with a new column 'dynamic_target'
    '''
    
    return data.with_columns([
        (pl.col('volatility_measure') * target_volatility_multiplier * pl.col('regime_multiplier'))
        .clip(base_min_breakout * 0.6, base_min_breakout * 1.4)
        .alias('dynamic_target')
    ])