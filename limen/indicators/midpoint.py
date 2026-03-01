import polars as pl


def midpoint(data: pl.DataFrame,
             col: str = 'close',
             period: int = 14) -> pl.DataFrame:

    '''
    Compute Midpoint Over Period (MIDPOINT) indicator.

    Equivalent to TA-Lib MIDPOINT: (rolling_max + rolling_min) / 2
    over a lookback window of `period` bars.

    Args:
        data (pl.DataFrame): Klines dataset with price column
        col (str): Column name for price data
        period (int): Number of periods for the rolling window

    Returns:
        pl.DataFrame: The input data with a new column 'midpoint_{period}'
    '''

    return data.with_columns([
        (
            (pl.col(col).rolling_max(window_size=period) + pl.col(col).rolling_min(window_size=period)) / 2
        ).alias(f"midpoint_{period}")
    ])
