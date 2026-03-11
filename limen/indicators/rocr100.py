import polars as pl


CMP_N_100000 = 100000

def rocr100(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 10,
) -> pl.DataFrame:

    '''
    Compute Rate of Change Ratio 100 scale (ROCR100): (price / prev_price) * 100.

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods (1..100000)

    Returns:
        pl.DataFrame: The input data with a new column 'rocr100_{period}'
    '''

    if period < 1 or period > CMP_N_100000:
        raise ValueError('period must be between 1 and 100000')

    out_col = f"rocr100_{period}"
    trailing = pl.col(price_col).shift(period)
    rocr100_expr = (
        pl.when(pl.int_range(0, pl.len()) < period)
        .then(None)
        .when(trailing != 0.0)
        .then((pl.col(price_col) / trailing) * 100.0)
        .otherwise(0.0)
        .alias(out_col)
    )
    return data.with_columns(rocr100_expr)
