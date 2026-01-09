import polars as pl
from limen.indicators.sma import sma


def volume_ratio(data: pl.DataFrame, period: int = 20) -> pl.DataFrame:
    
    '''
    Compute volume ratio relative to simple moving average baseline.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'volume' column
        period (int): Number of periods for SMA calculation
        
    Returns:
        pl.DataFrame: The input data with a new column 'volume_ratio'
    '''
    
    # Calculate volume SMA if not already present
    volume_sma_col = f'volume_sma_{period}'
    if volume_sma_col not in data.collect_schema().names():
        data = sma(data, 'volume', period)
    
    return data.with_columns([
        (pl.col('volume') / pl.col(volume_sma_col))
        .alias('volume_ratio')
    ])