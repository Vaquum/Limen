<<<<<<< HEAD
import numpy as np
import polars as pl


def midprice(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    period: int = 14,
) -> pl.DataFrame:

    '''
    Compute Midpoint Price over period.

    Args:
        data (pl.DataFrame): Dataset with high and low columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        period (int): Number of periods (2..100000)
=======
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
>>>>>>> origin/main

    Returns:
        pl.DataFrame: The input data with a new column 'midprice_{period}'
    '''

<<<<<<< HEAD
    if period < 2 or period > 100000:
        raise ValueError('period must be between 2 and 100000')

    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    n = len(data)
    out_col = f'midprice_{period}'
    out = np.full(n, np.nan, dtype=float)

    lookback_total = period - 1
    if n <= lookback_total:
        return data.with_columns(pl.Series(name=out_col, values=out))

    today = lookback_total
    trailing_idx = 0
    while today < n:
        lowest = low_values[trailing_idx]
        highest = high_values[trailing_idx]
        i = trailing_idx + 1
        while i <= today:
            low_tmp = low_values[i]
            if low_tmp < lowest:
                lowest = low_tmp
            high_tmp = high_values[i]
            if high_tmp > highest:
                highest = high_tmp
            i += 1

        out[today] = (highest + lowest) / 2.0
        today += 1
        trailing_idx += 1

    return data.with_columns(pl.Series(name=out_col, values=out))
=======
    return data.with_columns([
        (
            (pl.col(high_col).rolling_max(window_size=period) + pl.col(low_col).rolling_min(window_size=period)) / 2
        ).alias(f"midprice_{period}")
    ])
>>>>>>> origin/main
