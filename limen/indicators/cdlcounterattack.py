import numpy as np
import polars as pl

CDLCOUNTERATTACK_BODY_LONG_AVG_PERIOD = 10
CDLCOUNTERATTACK_EQUAL_AVG_PERIOD = 5
CDLCOUNTERATTACK_EQUAL_FACTOR = 0.05
CDLCOUNTERATTACK_EQUAL_PERIOD_TOTAL = 0.0


def cdlcounterattack(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Counterattack candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdlcounterattack'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    body_long_avg_period = CDLCOUNTERATTACK_BODY_LONG_AVG_PERIOD
    equal_avg_period = CDLCOUNTERATTACK_EQUAL_AVG_PERIOD
    equal_factor = CDLCOUNTERATTACK_EQUAL_FACTOR
    lookback_total = max(equal_avg_period, body_long_avg_period) + 1

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlcounterattack', values=out))

    equal_period_total = CDLCOUNTERATTACK_EQUAL_PERIOD_TOTAL
    body_long_period_total = np.zeros(2, dtype=float)
    equal_trailing_idx = lookback_total - equal_avg_period
    body_long_trailing_idx = lookback_total - body_long_avg_period

    i = equal_trailing_idx
    while i < lookback_total:
        equal_period_total += high_values[i - 1] - low_values[i - 1]
        i += 1

    i = body_long_trailing_idx
    while i < lookback_total:
        body_long_period_total[1] += abs(close_values[i - 1] - open_values[i - 1])
        body_long_period_total[0] += abs(close_values[i] - open_values[i])
        i += 1

    i = lookback_total
    while i < n:
        real_body_i1 = abs(close_values[i - 1] - open_values[i - 1])
        real_body_i0 = abs(close_values[i] - open_values[i])
        color_i1 = 1 if close_values[i - 1] >= open_values[i - 1] else -1
        color_i0 = 1 if close_values[i] >= open_values[i] else -1

        body_long_avg_i1 = body_long_period_total[1] / body_long_avg_period
        body_long_avg_i0 = body_long_period_total[0] / body_long_avg_period
        equal_avg_i1 = equal_factor * (equal_period_total / equal_avg_period)

        if (
            color_i1 == -color_i0
            and real_body_i1 > body_long_avg_i1
            and real_body_i0 > body_long_avg_i0
            and close_values[i] <= close_values[i - 1] + equal_avg_i1
            and close_values[i] >= close_values[i - 1] - equal_avg_i1
        ):
            out[i] = color_i0 * 100

        equal_period_total += (
            (high_values[i - 1] - low_values[i - 1])
            - (high_values[equal_trailing_idx - 1] - low_values[equal_trailing_idx - 1])
        )

        for tot_idx in range(1, -1, -1):
            body_long_period_total[tot_idx] += (
                abs(close_values[i - tot_idx] - open_values[i - tot_idx])
                - abs(close_values[body_long_trailing_idx - tot_idx] - open_values[body_long_trailing_idx - tot_idx])
            )

        i += 1
        equal_trailing_idx += 1
        body_long_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdlcounterattack', values=out))
