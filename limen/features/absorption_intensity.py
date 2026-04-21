import polars as pl

EPSILON = 1e-10


def absorption_intensity(
    data: pl.DataFrame,
    window: int = 20,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    volume_col: str = 'volume',
) -> pl.DataFrame:

    '''
    Compute high-effort low-result absorption intensity from volume and candle body share.

    Args:
        data (pl.DataFrame): Klines dataset with open, high, low, close, and volume columns
        window (int): Number of periods used for the trailing volume baseline
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices
        volume_col (str): Column name for traded volume

    Returns:
        pl.DataFrame: The input data with a new column 'absorption_intensity'
    '''

    body_to_range_expr = (pl.col(close_col) - pl.col(open_col)).abs() / ((pl.col(high_col) - pl.col(low_col)) + EPSILON)
    volume_baseline = pl.col(volume_col).rolling_mean(window_size=window).shift(1)

    return data.with_columns(
        (
            (pl.col(volume_col) / (volume_baseline + EPSILON))
            * (1.0 - body_to_range_expr)
        ).alias('absorption_intensity')
    )
