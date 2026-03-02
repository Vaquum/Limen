import numpy as np
import polars as pl


def cdlhikkakemod(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Modified Hikkake candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdlhikkakemod'
    '''

    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)

    # TA-Lib default candle setting for Near:
    # rangeType=HighLow, avgPeriod=5, factor=0.2
    near_avg_period = 5
    near_factor = 0.2
    lookback_total = max(1, near_avg_period) + 5

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlhikkakemod', values=out))

    near_period_total = 0.0
    near_trailing_idx = lookback_total - 3 - near_avg_period

    i = near_trailing_idx
    while i < lookback_total - 3:
        near_period_total += high_values[i - 2] - low_values[i - 2]
        i += 1

    pattern_idx = 0
    pattern_result = 0

    i = lookback_total - 3
    while i < lookback_total:
        near_avg_i2 = near_factor * (near_period_total / near_avg_period)
        bull_pattern = (
            high_values[i] < high_values[i - 1]
            and low_values[i] < low_values[i - 1]
            and close_values[i - 2] <= low_values[i - 2] + near_avg_i2
        )
        bear_pattern = (
            high_values[i] > high_values[i - 1]
            and low_values[i] > low_values[i - 1]
            and close_values[i - 2] >= high_values[i - 2] - near_avg_i2
        )

        if (
            high_values[i - 2] < high_values[i - 3]
            and low_values[i - 2] > low_values[i - 3]
            and high_values[i - 1] < high_values[i - 2]
            and low_values[i - 1] > low_values[i - 2]
            and (bull_pattern or bear_pattern)
        ):
            pattern_result = 100 * (1 if high_values[i] < high_values[i - 1] else -1)
            pattern_idx = i
        elif (
            i <= pattern_idx + 3
            and (
                (pattern_result > 0 and close_values[i] > high_values[pattern_idx - 1])
                or (pattern_result < 0 and close_values[i] < low_values[pattern_idx - 1])
            )
        ):
            pattern_idx = 0

        near_period_total += (
            (high_values[i - 2] - low_values[i - 2])
            - (high_values[near_trailing_idx - 2] - low_values[near_trailing_idx - 2])
        )
        near_trailing_idx += 1
        i += 1

    i = lookback_total
    while i < n:
        near_avg_i2 = near_factor * (near_period_total / near_avg_period)
        bull_pattern = (
            high_values[i] < high_values[i - 1]
            and low_values[i] < low_values[i - 1]
            and close_values[i - 2] <= low_values[i - 2] + near_avg_i2
        )
        bear_pattern = (
            high_values[i] > high_values[i - 1]
            and low_values[i] > low_values[i - 1]
            and close_values[i - 2] >= high_values[i - 2] - near_avg_i2
        )

        if (
            high_values[i - 2] < high_values[i - 3]
            and low_values[i - 2] > low_values[i - 3]
            and high_values[i - 1] < high_values[i - 2]
            and low_values[i - 1] > low_values[i - 2]
            and (bull_pattern or bear_pattern)
        ):
            pattern_result = 100 * (1 if high_values[i] < high_values[i - 1] else -1)
            pattern_idx = i
            out[i] = pattern_result
        elif (
            i <= pattern_idx + 3
            and (
                (pattern_result > 0 and close_values[i] > high_values[pattern_idx - 1])
                or (pattern_result < 0 and close_values[i] < low_values[pattern_idx - 1])
            )
        ):
            out[i] = pattern_result + (100 if pattern_result > 0 else -100)
            pattern_idx = 0

        near_period_total += (
            (high_values[i - 2] - low_values[i - 2])
            - (high_values[near_trailing_idx - 2] - low_values[near_trailing_idx - 2])
        )
        near_trailing_idx += 1
        i += 1

    return data.with_columns(pl.Series(name='cdlhikkakemod', values=out))
