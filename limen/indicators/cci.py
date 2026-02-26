import numpy as np
import polars as pl


def cci(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    period: int = 14,
) -> pl.DataFrame:

    '''
    Compute Commodity Channel Index (CCI).

    Args:
        data (pl.DataFrame): Dataset with high/low/close columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices
        period (int): Number of periods (2..100000)

    Returns:
        pl.DataFrame: The input data with a new column 'cci_{period}'
    '''

    if period < 2 or period > 100000:
        raise ValueError('period must be between 2 and 100000')

    high = data[high_col].to_numpy().astype(float, copy=False)
    low = data[low_col].to_numpy().astype(float, copy=False)
    close = data[close_col].to_numpy().astype(float, copy=False)

    n = len(data)
    out_col = f'cci_{period}'
    out = np.full(n, np.nan, dtype=float)

    lookback = period - 1
    if n <= lookback:
        return data.with_columns(pl.Series(name=out_col, values=out))

    typical_price = (high + low + close) / 3.0
    start_idx = lookback

    for i in range(start_idx, n):
        trailing = i - lookback
        average = 0.0
        for j in range(trailing, i + 1):
            average += typical_price[j]
        average /= period

        abs_sum = 0.0
        for j in range(trailing, i + 1):
            abs_sum += abs(typical_price[j] - average)

        diff = typical_price[i] - average
        if diff != 0.0 and abs_sum != 0.0:
            out[i] = diff / (0.015 * (abs_sum / period))
        else:
            out[i] = 0.0

    return data.with_columns(pl.Series(name=out_col, values=out))
