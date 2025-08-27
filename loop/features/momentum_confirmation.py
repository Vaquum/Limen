import polars as pl


def momentum_confirmation(df: pl.DataFrame, 
                         short_period: int = 1,
                         long_period: int = 3,
                         short_weight: float = 0.5) -> pl.DataFrame:
    
    '''
    Compute simple momentum confirmation scores based on recent price changes.
    
    Args:
        df (pl.DataFrame): Klines dataset with 'close' column
        short_period (int): Number of periods for short-term momentum
        long_period (int): Number of periods for long-term momentum
        short_weight (float): Weight for short-term momentum in final score
        
    Returns:
        pl.DataFrame: The input data with new columns 'momentum_1', 'momentum_3', 'momentum_score'
    '''
    
    if short_period >= long_period:
        raise ValueError(f"short_period ({short_period}) must be less than long_period ({long_period})")
    
    long_weight = 1.0 - short_weight
    
    df = df.with_columns([
        pl.col('close').pct_change(short_period).alias(f'momentum_{short_period}'),
        pl.col('close').pct_change(long_period).alias(f'momentum_{long_period}')
    ])
    
    df = df.with_columns([
        ((pl.col(f'momentum_{short_period}') > 0).cast(pl.Float32) * short_weight +
         (pl.col(f'momentum_{long_period}') > 0).cast(pl.Float32) * long_weight)
        .alias('momentum_score')
    ])
    
    return df