import polars as pl


EPSILON = 1e-10


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(denominator.is_null() | (denominator.abs() <= EPSILON))
        .then(None)
        .otherwise(numerator / denominator)
    )


def _true_range(
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.Expr:
    previous_close = pl.col(close_col).shift(1)

    return pl.max_horizontal(
        pl.col(high_col) - pl.col(low_col),
        (pl.col(high_col) - previous_close).abs(),
        (pl.col(low_col) - previous_close).abs(),
    )


def close_ma_distance_atr(
    data: pl.DataFrame,
    ma_window: int = 20,
    atr_window: int = 14,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    output_col: str = 'close_ma_distance_atr',
) -> pl.DataFrame:

    '''
    Compute close-to-SMA distance normalized by SMA-smoothed true range.

    Args:
        data (pl.DataFrame): Klines dataset with high, low, and close columns
        ma_window (int): Number of periods in the simple moving average
        atr_window (int): Number of periods in the true-range moving average
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with the ATR-normalized distance column appended
    '''

    moving_average = pl.col(close_col).rolling_mean(window_size=ma_window)
    smoothed_range = _true_range(high_col, low_col, close_col).rolling_mean(
        window_size=atr_window
    )

    return data.with_columns(
        _safe_divide(pl.col(close_col) - moving_average, smoothed_range).alias(output_col)
    )
