import numpy as np
import polars as pl

CDLIDENTICAL3CROWS_EQUAL_AVG_PERIOD = 5
CDLIDENTICAL3CROWS_EQUAL_FACTOR = 0.05
CDLIDENTICAL3CROWS_SHADOW_VS_AVG_PERIOD = 10
CDLIDENTICAL3CROWS_SHADOW_VS_FACTOR = 0.1


def cdlidentical3crows(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Identical Three Crows candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdlidentical3crows'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    shadow_vs_avg_period = CDLIDENTICAL3CROWS_SHADOW_VS_AVG_PERIOD
    shadow_vs_factor = CDLIDENTICAL3CROWS_SHADOW_VS_FACTOR
    equal_avg_period = CDLIDENTICAL3CROWS_EQUAL_AVG_PERIOD
    equal_factor = CDLIDENTICAL3CROWS_EQUAL_FACTOR
    lookback_total = max(shadow_vs_avg_period, equal_avg_period) + 2

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlidentical3crows', values=out))

    shadow_vs_period_total = np.zeros(3, dtype=float)
    equal_period_total = np.zeros(3, dtype=float)
    shadow_vs_trailing_idx = lookback_total - shadow_vs_avg_period
    equal_trailing_idx = lookback_total - equal_avg_period

    i = shadow_vs_trailing_idx
    while i < lookback_total:
        shadow_vs_period_total[2] += high_values[i - 2] - low_values[i - 2]
        shadow_vs_period_total[1] += high_values[i - 1] - low_values[i - 1]
        shadow_vs_period_total[0] += high_values[i] - low_values[i]
        i += 1

    i = equal_trailing_idx
    while i < lookback_total:
        equal_period_total[2] += high_values[i - 2] - low_values[i - 2]
        equal_period_total[1] += high_values[i - 1] - low_values[i - 1]
        i += 1

    i = lookback_total
    while i < n:
        color_i2 = 1 if close_values[i - 2] >= open_values[i - 2] else -1
        color_i1 = 1 if close_values[i - 1] >= open_values[i - 1] else -1
        color_i0 = 1 if close_values[i] >= open_values[i] else -1

        lower_shadow_i2 = min(open_values[i - 2], close_values[i - 2]) - low_values[i - 2]
        lower_shadow_i1 = min(open_values[i - 1], close_values[i - 1]) - low_values[i - 1]
        lower_shadow_i0 = min(open_values[i], close_values[i]) - low_values[i]

        shadow_vs_avg_i2 = shadow_vs_factor * (shadow_vs_period_total[2] / shadow_vs_avg_period)
        shadow_vs_avg_i1 = shadow_vs_factor * (shadow_vs_period_total[1] / shadow_vs_avg_period)
        shadow_vs_avg_i0 = shadow_vs_factor * (shadow_vs_period_total[0] / shadow_vs_avg_period)
        equal_avg_i2 = equal_factor * (equal_period_total[2] / equal_avg_period)
        equal_avg_i1 = equal_factor * (equal_period_total[1] / equal_avg_period)

        if (
            color_i2 == -1
            and lower_shadow_i2 < shadow_vs_avg_i2
            and color_i1 == -1
            and lower_shadow_i1 < shadow_vs_avg_i1
            and color_i0 == -1
            and lower_shadow_i0 < shadow_vs_avg_i0
            and close_values[i - 2] > close_values[i - 1]
            and close_values[i - 1] > close_values[i]
            and open_values[i - 1] <= close_values[i - 2] + equal_avg_i2
            and open_values[i - 1] >= close_values[i - 2] - equal_avg_i2
            and open_values[i] <= close_values[i - 1] + equal_avg_i1
            and open_values[i] >= close_values[i - 1] - equal_avg_i1
        ):
            out[i] = -100

        for tot_idx in range(2, -1, -1):
            shadow_vs_period_total[tot_idx] += (
                (high_values[i - tot_idx] - low_values[i - tot_idx])
                - (high_values[shadow_vs_trailing_idx - tot_idx] - low_values[shadow_vs_trailing_idx - tot_idx])
            )
        for tot_idx in range(2, 0, -1):
            equal_period_total[tot_idx] += (
                (high_values[i - tot_idx] - low_values[i - tot_idx])
                - (high_values[equal_trailing_idx - tot_idx] - low_values[equal_trailing_idx - tot_idx])
            )

        i += 1
        shadow_vs_trailing_idx += 1
        equal_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdlidentical3crows', values=out))
