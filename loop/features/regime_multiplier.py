import polars as pl


def regime_multiplier(data: pl.DataFrame) -> pl.DataFrame:
    
    '''
    Compute volatility regime-based multiplier for dynamic parameter adjustment.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'volatility_regime' column
        
    Returns:
        pl.DataFrame: The input data with a new column 'regime_multiplier'
    '''
    
    return data.with_columns([
        pl.when(pl.col('volatility_regime') == 'low')
            .then(pl.lit(0.8))
            .when(pl.col('volatility_regime') == 'high')
            .then(pl.lit(1.2))
            .otherwise(pl.lit(1.0))
            .alias('regime_multiplier')
    ])