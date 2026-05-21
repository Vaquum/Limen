import polars as pl


EPSILON = 1e-10


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(denominator.is_null() | (denominator.abs() <= EPSILON))
        .then(None)
        .otherwise(numerator / denominator)
    )


def maker_liquidity_share(
    data: pl.DataFrame,
    maker_liquidity_col: str = 'maker_liquidity',
    liquidity_col: str = 'liquidity_sum',
    output_col: str = 'maker_liquidity_share',
) -> pl.DataFrame:

    '''
    Compute maker liquidity as a share of total liquidity.

    Args:
        data (pl.DataFrame): Klines dataset with maker and total liquidity columns
        maker_liquidity_col (str): Column name for maker-side liquidity
        liquidity_col (str): Column name for total liquidity
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with the maker-liquidity share column appended
    '''

    return data.with_columns(
        _safe_divide(pl.col(maker_liquidity_col), pl.col(liquidity_col)).alias(output_col)
    )
