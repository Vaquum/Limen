import polars as pl
from limen.indicators.sma import sma


def sma_ratios(data: pl.DataFrame, periods: list = [5, 10, 20, 50], price_col: str = 'close') -> pl.DataFrame:
    
    '''
    Compute price to SMA ratios for multiple time periods.
    
    Args:
        data (pl.DataFrame): Klines dataset with price column
        periods (list): List of SMA periods to calculate ratios for
        price_col (str): Name of the price column (default: 'close')
        
    Returns:
        pl.DataFrame: The input data with new columns 'sma_{period}_ratio'
    '''
    
    df = data.clone()
    
    for period in periods:
        sma_col = f'{price_col}_sma_{period}'
        
        # Calculate SMA if not already present
        if sma_col not in df.collect_schema().names():
            df = sma(df, price_col, period)
        
        # Add ratio column
        df = df.with_columns([
            (pl.col(price_col) / pl.col(sma_col)).alias(f'sma_{period}_ratio')
        ])
    
    return df