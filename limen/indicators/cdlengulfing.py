import numpy as np
import polars as pl


def cdlengulfing(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Engulfing candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdlengulfing'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)

    # TA-Lib lookback for CDLENGULFING.
    lookback_total = 2

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlengulfing', values=out))

    i = lookback_total
    while i < n:
        color_i = 1 if close_values[i] >= open_values[i] else -1
        color_i1 = 1 if close_values[i - 1] >= open_values[i - 1] else -1

        bullish_engulf = (
            color_i == 1
            and color_i1 == -1
            and (
                (close_values[i] >= open_values[i - 1] and open_values[i] < close_values[i - 1])
                or (close_values[i] > open_values[i - 1] and open_values[i] <= close_values[i - 1])
            )
        )
        bearish_engulf = (
            color_i == -1
            and color_i1 == 1
            and (
                (open_values[i] >= close_values[i - 1] and close_values[i] < open_values[i - 1])
                or (open_values[i] > close_values[i - 1] and close_values[i] <= open_values[i - 1])
            )
        )

        if bullish_engulf or bearish_engulf:
            if open_values[i] != close_values[i - 1] and close_values[i] != open_values[i - 1]:
                out[i] = color_i * 100
            else:
                out[i] = color_i * 80

        i += 1

    return data.with_columns(pl.Series(name='cdlengulfing', values=out))
