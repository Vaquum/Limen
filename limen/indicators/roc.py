import polars as pl


CMP_N_100000 = 100000

def roc(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 10,
) -> pl.DataFrame:

    '''
    Compute Rate of Change (ROC): ((price / prev_price) - 1) * 100.

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods (1..100000)

    Returns:
        pl.DataFrame: The input data with a new column 'roc_{period}'
    '''

    if period < 1 or period > CMP_N_100000:
        raise ValueError('period must be between 1 and 100000')

    out_col = f'roc_{period}'
    trailing = pl.col(price_col).shift(period)
    roc_expr = (
        pl.when(pl.int_range(0, pl.len()) < period)
        .then(None)
        .when(trailing != 0.0)
        .then(((pl.col(price_col) / trailing) - 1.0) * 100.0)
        .otherwise(0.0)
        .alias(out_col)
    )
    return data.with_columns(roc_expr)
