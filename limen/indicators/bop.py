import polars as pl


BOP_EPSILON = 1e-14


def bop(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Balance of Power (BOP).

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'bop'
    '''

    denominator = pl.col(high_col) - pl.col(low_col)
    bop_expr = (
        pl.when(denominator >= BOP_EPSILON)
        .then((pl.col(close_col) - pl.col(open_col)) / denominator)
        .otherwise(0.0)
        .alias('bop')
    )

    return data.with_columns(bop_expr)
