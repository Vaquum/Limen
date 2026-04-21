import polars as pl


def trend_coherence(
    data: pl.DataFrame,
    short_window: int = 12,
    medium_window: int = 48,
    long_window: int = 168,
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute directional agreement across short, medium, and long return horizons.

    Args:
        data (pl.DataFrame): Klines dataset with a close price column
        short_window (int): Short return horizon
        medium_window (int): Medium return horizon
        long_window (int): Long return horizon
        close_col (str): Column name used for return calculations

    Returns:
        pl.DataFrame: The input data with a new column 'trend_coherence'
    '''

    short_sign = pl.col(close_col).pct_change(short_window).sign()
    medium_sign = pl.col(close_col).pct_change(medium_window).sign()
    long_sign = pl.col(close_col).pct_change(long_window).sign()

    return data.with_columns(
        ((short_sign + medium_sign + long_sign) / 3.0).alias('trend_coherence')
    )
