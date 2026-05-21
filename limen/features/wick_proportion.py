import polars as pl


EPSILON = 1e-10


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(denominator.is_null() | (denominator.abs() <= EPSILON))
        .then(None)
        .otherwise(numerator / denominator)
    )


def wick_proportion(
    data: pl.DataFrame,
    window: int = 20,
    high_col: str = 'high',
    low_col: str = 'low',
    open_col: str = 'open',
    close_col: str = 'close',
    output_col: str = 'wick_proportion',
) -> pl.DataFrame:

    '''
    Compute rolling mean wick share of the full candle range.

    Args:
        data (pl.DataFrame): Klines dataset with open, high, low, and close columns
        window (int): Number of periods in the rolling mean
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        open_col (str): Column name for open prices
        close_col (str): Column name for close prices
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with the wick-proportion column appended
    '''

    price_range = pl.col(high_col) - pl.col(low_col)
    wick = price_range - (pl.col(close_col) - pl.col(open_col)).abs()

    return data.with_columns(
        _safe_divide(wick, price_range)
        .rolling_mean(window_size=window)
        .alias(output_col)
    )
