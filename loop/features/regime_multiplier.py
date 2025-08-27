import polars as pl

# Volatility regime multiplier constants
LOW_VOLATILITY_MULTIPLIER = 0.8    # Reduce targets/stops in low volatility
NORMAL_VOLATILITY_MULTIPLIER = 1.0  # Baseline multiplier for normal volatility
HIGH_VOLATILITY_MULTIPLIER = 1.2   # Increase targets/stops in high volatility


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
            .then(pl.lit(LOW_VOLATILITY_MULTIPLIER))
            .when(pl.col('volatility_regime') == 'high')
            .then(pl.lit(HIGH_VOLATILITY_MULTIPLIER))
            .otherwise(pl.lit(NORMAL_VOLATILITY_MULTIPLIER))
            .alias('regime_multiplier')
    ])