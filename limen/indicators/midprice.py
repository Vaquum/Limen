import polars as pl


def midprice(data: pl.DataFrame,
             high_col: str = 'high',
             low_col: str = 'low',
             period: int = 14) -> pl.DataFrame:

    '''
    Compute Midpoint Price Over Period (MIDPRICE) indicator.

    Equivalent to TA-Lib MIDPRICE: (rolling_max(high, period) + rolling_min(low, period)) / 2
    over a lookback window of `period` bars.

    Args:
        data (pl.DataFrame): Klines dataset with 'high' and 'low' columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        period (int): Number of periods for the rolling window

    Returns:
        pl.DataFrame: The input data with a new column 'midprice_{period}'
    '''

    return data.with_columns([
        (
            (pl.col(high_col).rolling_max(window_size=period) + pl.col(low_col).rolling_min(window_size=period)) / 2
        ).alias(f"midprice_{period}")
    ])
