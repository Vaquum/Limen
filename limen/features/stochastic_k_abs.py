import polars as pl


EPSILON = 1e-10


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(denominator.is_null() | (denominator.abs() <= EPSILON))
        .then(None)
        .otherwise(numerator / denominator)
    )


def stochastic_k_abs(
    data: pl.DataFrame,
    window: int = 14,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    output_col: str = 'stochastic_k_abs',
) -> pl.DataFrame:

    '''
    Compute absolute distance of stochastic percent K from the center line.

    Args:
        data (pl.DataFrame): Klines dataset with high, low, and close columns
        window (int): Number of periods in the high-low channel
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with the stochastic distance column appended
    '''

    rolling_low = pl.col(low_col).rolling_min(window_size=window)
    rolling_high = pl.col(high_col).rolling_max(window_size=window)
    stoch_k = _safe_divide(pl.col(close_col) - rolling_low, rolling_high - rolling_low)

    return data.with_columns((stoch_k - 0.5).abs().alias(output_col))
