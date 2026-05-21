import polars as pl


EPSILON = 1e-10


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(denominator.is_null() | (denominator.abs() <= EPSILON))
        .then(None)
        .otherwise(numerator / denominator)
    )


def downside_volatility_ratio(
    data: pl.DataFrame,
    window: int = 20,
    close_col: str = 'close',
    output_col: str = 'downside_volatility_ratio',
) -> pl.DataFrame:

    '''
    Compute rolling downside squared-return share of total squared returns.

    Args:
        data (pl.DataFrame): Klines dataset with a close price column
        window (int): Number of periods in the rolling sums
        close_col (str): Column name for close prices
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with the downside-volatility ratio appended
    '''

    returns = pl.col(close_col).pct_change()
    squared = returns ** 2
    downside = pl.when(returns < 0.0).then(squared).otherwise(0.0)

    return data.with_columns(
        _safe_divide(
            downside.rolling_sum(window_size=window),
            squared.rolling_sum(window_size=window),
        ).alias(output_col)
    )
