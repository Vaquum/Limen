import polars as pl


EPSILON = 1e-10


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(denominator.is_null() | (denominator.abs() <= EPSILON))
        .then(None)
        .otherwise(numerator / denominator)
    )


def trade_size_ratio(
    data: pl.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
    volume_col: str = 'volume',
    trades_col: str = 'no_of_trades',
    output_col: str = 'trade_size_ratio',
) -> pl.DataFrame:

    '''
    Compare short and long rolling average trade sizes.

    Args:
        data (pl.DataFrame): Klines dataset with volume and trade-count columns
        short_window (int): Number of periods in the short average
        long_window (int): Number of periods in the long average
        volume_col (str): Column name for total traded volume
        trades_col (str): Column name for number of trades
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with the trade-size ratio column appended
    '''

    trade_size = _safe_divide(pl.col(volume_col), pl.col(trades_col))

    return data.with_columns(
        _safe_divide(
            trade_size.rolling_mean(window_size=short_window),
            trade_size.rolling_mean(window_size=long_window),
        ).alias(output_col)
    )
