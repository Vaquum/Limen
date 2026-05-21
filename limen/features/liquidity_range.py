import polars as pl


EPSILON = 1e-10


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(denominator.is_null() | (denominator.abs() <= EPSILON))
        .then(None)
        .otherwise(numerator / denominator)
    )


def liquidity_range(
    data: pl.DataFrame,
    window: int = 20,
    high_liquidity_col: str = 'high_liquidity',
    low_liquidity_col: str = 'low_liquidity',
    output_col: str = 'liquidity_range',
) -> pl.DataFrame:

    '''
    Compute rolling mean high-liquidity to low-liquidity ratio.

    Args:
        data (pl.DataFrame): Klines dataset with high- and low-liquidity columns
        window (int): Number of periods in the rolling mean
        high_liquidity_col (str): Column name for high-side liquidity
        low_liquidity_col (str): Column name for low-side liquidity
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with the liquidity-range column appended
    '''

    return data.with_columns(
        _safe_divide(pl.col(high_liquidity_col), pl.col(low_liquidity_col))
        .rolling_mean(window_size=window)
        .alias(output_col)
    )
