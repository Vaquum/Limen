import polars as pl


def momentum_weight(data: pl.DataFrame, period: int = 12) -> pl.DataFrame:
    
    '''
    Compute momentum-based weighting factor from price change direction.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'close' column
        period (int): Period for momentum calculation
        
    Returns:
        pl.DataFrame: The input data with a new column 'momentum_weight'
    '''
    
    return data.with_columns([
        ((pl.col('close').pct_change(period) > 0).cast(pl.Float32) * 0.5 + 0.5)
        .alias('momentum_weight')
    ])