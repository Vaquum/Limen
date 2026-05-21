import polars as pl


EPSILON = 1e-10


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(denominator.is_null() | (denominator.abs() <= EPSILON))
        .then(None)
        .otherwise(numerator / denominator)
    )


def taker_imbalance_ratio(
    data: pl.DataFrame,
    window: int = 20,
    volume_col: str = 'volume',
    maker_ratio_col: str = 'maker_ratio',
    output_col: str = 'taker_imbalance_ratio',
) -> pl.DataFrame:

    '''
    Compute rolling absolute taker imbalance as a share of volume.

    Args:
        data (pl.DataFrame): Klines dataset with volume and maker-ratio columns
        window (int): Number of periods in the rolling mean
        volume_col (str): Column name for total traded volume
        maker_ratio_col (str): Column name for maker-side volume ratio
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with the taker-imbalance ratio column appended
    '''

    imbalance = (pl.col(volume_col) * (1.0 - (2.0 * pl.col(maker_ratio_col)))).abs()

    return data.with_columns(
        _safe_divide(imbalance, pl.col(volume_col))
        .rolling_mean(window_size=window)
        .alias(output_col)
    )
