import polars as pl


CMP_N_100000 = 100000

def mom(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 10,
) -> pl.DataFrame:

    '''
    Compute Momentum (MOM).

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods (1..100000)

    Returns:
        pl.DataFrame: The input data with a new column 'mom_{period}'
    '''

    if period < 1 or period > CMP_N_100000:
        raise ValueError('mom period must be between 1 and 100000')

    out_col = f"mom_{period}"
    mom_expr = (
        pl.when(pl.int_range(0, pl.len()) < period)
        .then(None)
        .otherwise(pl.col(price_col) - pl.col(price_col).shift(period))
        .alias(out_col)
    )
    return data.with_columns(mom_expr)
