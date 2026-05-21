import polars as pl


EPSILON = 1e-10


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(denominator.is_null() | (denominator.abs() <= EPSILON))
        .then(None)
        .otherwise(numerator / denominator)
    )


def liquidity_drop(
    data: pl.DataFrame,
    window: int = 20,
    liquidity_col: str = 'liquidity_sum',
    output_col: str = 'liquidity_drop',
) -> pl.DataFrame:

    '''
    Compute current liquidity relative to liquidity from a fixed lag.

    Args:
        data (pl.DataFrame): Klines dataset with a total-liquidity column
        window (int): Lag used as the denominator
        liquidity_col (str): Column name for total liquidity
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with the liquidity-drop column appended
    '''

    liquidity = pl.col(liquidity_col)

    return data.with_columns(
        _safe_divide(liquidity, liquidity.shift(window)).alias(output_col)
    )
