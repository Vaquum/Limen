import polars as pl



def regime_multiplier(data: pl.DataFrame,
                     regime_low_volatility_multiplier: float = 0.8,
                     regime_normal_volatility_multiplier: float = 1.0,
                     regime_high_volatility_multiplier: float = 1.2) -> pl.DataFrame:
    
    '''
    Compute volatility regime-based multiplier for dynamic parameter adjustment.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'volatility_regime' column
        regime_low_volatility_multiplier (float): Multiplier for low volatility regime
        regime_normal_volatility_multiplier (float): Multiplier for normal volatility regime
        regime_high_volatility_multiplier (float): Multiplier for high volatility regime
        
    Returns:
        pl.DataFrame: The input data with a new column 'regime_multiplier'
    '''
    
    return data.with_columns([
        pl.when(pl.col('volatility_regime') == 'low')
            .then(pl.lit(regime_low_volatility_multiplier))
            .when(pl.col('volatility_regime') == 'high')
            .then(pl.lit(regime_high_volatility_multiplier))
            .otherwise(pl.lit(regime_normal_volatility_multiplier))
            .alias('regime_multiplier')
    ])