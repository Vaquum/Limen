import polars as pl

def roc(data: pl.DataFrame,
        col: str = "close",
        period: int = 12) -> pl.DataFrame:
    
    '''
    Compute Rate of Change (ROC) over `period` for `col` and append as 'roc'.

    Args:
        data (pl.DataFrame): The input DataFrame.
        col (str): The column name on which to compute ROC.
        period (int): The look‐back period for ROC calculation.

    Returns:
        pl.DataFrame: The input data with the ROC column appended.
    '''
    
    prior = pl.col(col).shift(period)
    roc_expr = ((pl.col(col) - prior) / prior * 100).alias("roc")

    return data.with_columns([roc_expr])
