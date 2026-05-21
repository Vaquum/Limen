import polars as pl


EPSILON = 1e-10


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(denominator.is_null() | (denominator.abs() <= EPSILON))
        .then(None)
        .otherwise(numerator / denominator)
    )


def trade_density(
    data: pl.DataFrame,
    window: int = 20,
    trades_col: str = 'no_of_trades',
    volume_col: str = 'volume',
    output_col: str = 'trade_density',
) -> pl.DataFrame:

    '''
    Compute rolling mean trade count per unit of volume.

    Args:
        data (pl.DataFrame): Klines dataset with trade-count and volume columns
        window (int): Number of periods in the rolling mean
        trades_col (str): Column name for number of trades
        volume_col (str): Column name for total traded volume
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with the trade-density column appended
    '''

    return data.with_columns(
        _safe_divide(pl.col(trades_col), pl.col(volume_col))
        .rolling_mean(window_size=window)
        .alias(output_col)
    )
