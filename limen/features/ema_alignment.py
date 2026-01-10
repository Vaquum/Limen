import polars as pl


def ema_alignment(data: pl.DataFrame, ema_span: int = 21, power: float = 2.0) -> pl.DataFrame:
    
    '''
    Compute EMA alignment score with power transformation.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'close' column
        ema_span (int): Span for EMA calculation
        power (float): Power for alignment transformation
        
    Returns:
        pl.DataFrame: The input data with new columns 'ema', 'ema_alignment'
    '''
    
    df = data.with_columns([
        pl.col('close').ewm_mean(span=ema_span, adjust=False).alias('ema')
    ])
    
    df = df.with_columns([
        (1 - (pl.col('close') - pl.col('ema')).abs() / pl.col('ema'))
        .clip(0, 1)
        .pow(power)
        .alias('ema_alignment')
    ])
    
    return df