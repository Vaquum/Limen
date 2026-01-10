import polars as pl


def log_returns(data: pl.DataFrame, price_col: str = 'close') -> pl.DataFrame:
    
    '''
    Compute logarithmic returns for price series.
    
    Args:
        data (pl.DataFrame): Dataset with price column
        price_col (str): Name of the price column (default: 'close')
        
    Returns:
        pl.DataFrame: The input data with a new column 'log_returns'
    '''
    
    return data.with_columns([
        (pl.col(price_col) / pl.col(price_col).shift(1)).log().alias('log_returns')
    ])