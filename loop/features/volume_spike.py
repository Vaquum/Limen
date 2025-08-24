import polars as pl


def volume_spike(data: pl.DataFrame, period: int = 20) -> pl.DataFrame:
    
    '''
    Compute volume spike relative to rolling mean baseline.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'volume' column
        period (int): Number of periods for rolling mean calculation
        
    Returns:
        pl.DataFrame: The input data with a new column 'volume_spike'
    '''
    
    return data.with_columns([
        (pl.col('volume') / pl.col('volume').rolling_mean(window_size=period))
        .alias('volume_spike')
    ])