import numpy as np
import polars as pl

WMA_PERIOD_SUB = 0.0
WMA_PERIOD_SUM = 0.0


def wma(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 30,
) -> pl.DataFrame:

    '''
    Compute Weighted Moving Average (WMA).

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods

    Returns:
        pl.DataFrame: The input data with a new column 'wma_{period}'
    '''

    if period < 2 or period > 100000:
        raise ValueError('period must be between 2 and 100000')

    values = data[price_col].to_numpy().astype(float, copy=False)
    n = len(values)
    out_col = f'wma_{period}'
    out = np.full(n, np.nan, dtype=float)

    lookback_total = period - 1
    if n <= lookback_total:
        return data.with_columns(pl.Series(name=out_col, values=out))

    start_idx = lookback_total
    end_idx = n - 1

    divider = (period * (period + 1)) // 2

    out_idx = start_idx
    trailing_idx = start_idx - lookback_total

    period_sum = WMA_PERIOD_SUM
    period_sub = WMA_PERIOD_SUB
    in_idx = trailing_idx
    i = 1
    while in_idx < start_idx:
        temp_real = values[in_idx]
        in_idx += 1
        period_sub += temp_real
        period_sum += temp_real * i
        i += 1

    trailing_value = 0.0

    while in_idx <= end_idx:
        temp_real = values[in_idx]
        in_idx += 1

        period_sub += temp_real
        period_sub -= trailing_value
        period_sum += temp_real * period

        trailing_value = values[trailing_idx]
        trailing_idx += 1

        out[out_idx] = period_sum / divider
        out_idx += 1

        period_sum -= period_sub

    return data.with_columns(pl.Series(name=out_col, values=out))
