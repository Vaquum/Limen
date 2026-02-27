import numpy as np
import polars as pl


def willr(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    period: int = 14,
) -> pl.DataFrame:

    '''
    Compute Williams' %R (WILLR).

    Args:
        data (pl.DataFrame): Dataset with high/low/close columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices
        period (int): Number of periods (2..100000)

    Returns:
        pl.DataFrame: The input data with a new column 'willr_{period}'
    '''

    if period < 2 or period > 100000:
        raise ValueError('period must be between 2 and 100000')

    high = data[high_col].to_numpy().astype(float, copy=False)
    low = data[low_col].to_numpy().astype(float, copy=False)
    close = data[close_col].to_numpy().astype(float, copy=False)
    n = len(close)

    out_col = f'willr_{period}'
    out = np.full(n, np.nan, dtype=float)

    lookback = period - 1
    start_idx = lookback
    if start_idx >= n:
        return data.with_columns(pl.Series(name=out_col, values=out))

    today = start_idx
    trailing_idx = start_idx - lookback
    lowest_idx = -1
    highest_idx = -1
    lowest = 0.0
    highest = 0.0
    diff = 0.0

    while today < n:
        tmp = low[today]
        if lowest_idx < trailing_idx:
            lowest_idx = trailing_idx
            lowest = low[lowest_idx]
            i = lowest_idx + 1
            while i <= today:
                tmp = low[i]
                if tmp < lowest:
                    lowest_idx = i
                    lowest = tmp
                i += 1
            diff = (highest - lowest) / (-100.0)
        elif tmp <= lowest:
            lowest_idx = today
            lowest = tmp
            diff = (highest - lowest) / (-100.0)

        tmp = high[today]
        if highest_idx < trailing_idx:
            highest_idx = trailing_idx
            highest = high[highest_idx]
            i = highest_idx + 1
            while i <= today:
                tmp = high[i]
                if tmp > highest:
                    highest_idx = i
                    highest = tmp
                i += 1
            diff = (highest - lowest) / (-100.0)
        elif tmp >= highest:
            highest_idx = today
            highest = tmp
            diff = (highest - lowest) / (-100.0)

        if diff != 0.0:
            out[today] = (highest - close[today]) / diff
        else:
            out[today] = 0.0

        trailing_idx += 1
        today += 1

    return data.with_columns(pl.Series(name=out_col, values=out))
