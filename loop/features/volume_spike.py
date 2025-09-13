import polars as pl


def volume_spike(data: pl.DataFrame, period: int = 20, use_zscore: bool = False) -> pl.DataFrame:
    
    '''
    Compute volume spike relative to rolling statistics baseline.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'volume' column
        period (int): Number of periods for rolling statistics calculation
        use_zscore (bool): If True, use z-score (incorporating std dev), otherwise use ratio to mean
        
    Returns:
        pl.DataFrame: The input data with a new column 'volume_spike'
    '''
    
    if use_zscore:
        return data.with_columns([
            ((pl.col('volume') - pl.col('volume').rolling_mean(window_size=period)) /
             (pl.col('volume').rolling_std(window_size=period) + 1e-10))
            .alias('volume_spike')
        ])
    else:
        return data.with_columns([
            (pl.col('volume') / pl.col('volume').rolling_mean(window_size=period))
            .alias('volume_spike')
        ])