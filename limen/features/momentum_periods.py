import polars as pl


def momentum_periods(data: pl.DataFrame, periods: list = [12, 24, 48], price_col: str = 'close') -> pl.DataFrame:
    
    '''
    Compute momentum over multiple time periods.
    
    Args:
        data (pl.DataFrame): Dataset with price column
        periods (list): List of periods for momentum calculation
        price_col (str): Name of the price column (default: 'close')
        
    Returns:
        pl.DataFrame: The input data with new columns 'momentum_{period}' for each period
    '''
    
    momentum_expressions = []
    for period in periods:
        momentum_expressions.append(
            pl.col(price_col).pct_change(period).alias(f'momentum_{period}')
        )
    
    return data.with_columns(momentum_expressions)