import polars as pl


def var(data: pl.DataFrame,
        col: str = 'close',
        period: int = 5) -> pl.DataFrame:

    '''
    Compute Variance (VAR) over a rolling period.

    Equivalent to TA-Lib VAR: rolling sample variance (ddof=1) over `period` bars.
    First `period - 1` values are null (TA-Lib lookback behaviour).

    Args:
        data (pl.DataFrame): Klines dataset with price column
        col (str): Column name for price data
        period (int): Number of periods for the rolling window

    Returns:
        pl.DataFrame: The input data with a new column 'var_{period}'
    '''

    return data.with_columns([
        pl.col(col).rolling_var(window_size=period).alias(f"var_{period}")
    ])
