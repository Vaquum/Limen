import polars as pl

def roc(data: pl.DataFrame,
        col: str = 'close',
        period: int = 12) -> pl.DataFrame:
    
    '''
    Compute Rate of Change (ROC) indicator as percentage change.

    Args:
        data (pl.DataFrame): Klines dataset with price column
        col (str): Column name for price data
        period (int): Number of periods for ROC calculation

    Returns:
        pl.DataFrame: The input data with a new column 'roc_{period}'
    '''
    
    prior = pl.col(col).shift(period)
    roc_expr = ((pl.col(col) - prior) / prior * 100).alias(f"roc_{period}")

    return data.with_columns([roc_expr])