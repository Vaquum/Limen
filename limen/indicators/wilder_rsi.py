import polars as pl

from limen.indicators.rsi import rsi


def wilder_rsi(data: pl.DataFrame,
               period: int = 14) -> pl.DataFrame:

    '''
    Compute Wilder's RSI using canonical Wilder smoothing.

    Args:
        data (pl.DataFrame): Klines dataset with 'close' column
        period (int): Number of periods for RSI calculation

    Returns:
        pl.DataFrame: The input data with a new column 'wilder_rsi_{period}'
    '''

    return rsi(data, price_col='close', period=period).rename(
        {f'rsi_{period}': f'wilder_rsi_{period}'}
    )
