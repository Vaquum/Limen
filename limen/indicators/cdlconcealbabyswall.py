import numpy as np
import polars as pl

CDLCONCEALBABYSWALL_SHADOW_VS_AVG_PERIOD = 10
CDLCONCEALBABYSWALL_SHADOW_VS_FACTOR = 0.1


def cdlconcealbabyswall(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Concealing Baby Swallow candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdlconcealbabyswall'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    shadow_vs_avg_period = CDLCONCEALBABYSWALL_SHADOW_VS_AVG_PERIOD
    shadow_vs_factor = CDLCONCEALBABYSWALL_SHADOW_VS_FACTOR
    lookback_total = shadow_vs_avg_period + 3

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlconcealbabyswall', values=out))

    shadow_vs_period_total = np.zeros(4, dtype=float)
    shadow_vs_trailing_idx = lookback_total - shadow_vs_avg_period

    i = shadow_vs_trailing_idx
    while i < lookback_total:
        shadow_vs_period_total[3] += high_values[i - 3] - low_values[i - 3]
        shadow_vs_period_total[2] += high_values[i - 2] - low_values[i - 2]
        shadow_vs_period_total[1] += high_values[i - 1] - low_values[i - 1]
        i += 1

    i = lookback_total
    while i < n:
        color_i3 = 1 if close_values[i - 3] >= open_values[i - 3] else -1
        color_i2 = 1 if close_values[i - 2] >= open_values[i - 2] else -1
        color_i1 = 1 if close_values[i - 1] >= open_values[i - 1] else -1
        color_i0 = 1 if close_values[i] >= open_values[i] else -1

        lower_shadow_i3 = min(open_values[i - 3], close_values[i - 3]) - low_values[i - 3]
        upper_shadow_i3 = high_values[i - 3] - max(open_values[i - 3], close_values[i - 3])
        lower_shadow_i2 = min(open_values[i - 2], close_values[i - 2]) - low_values[i - 2]
        upper_shadow_i2 = high_values[i - 2] - max(open_values[i - 2], close_values[i - 2])
        upper_shadow_i1 = high_values[i - 1] - max(open_values[i - 1], close_values[i - 1])

        shadow_vs_avg_i3 = shadow_vs_factor * (shadow_vs_period_total[3] / shadow_vs_avg_period)
        shadow_vs_avg_i2 = shadow_vs_factor * (shadow_vs_period_total[2] / shadow_vs_avg_period)
        shadow_vs_avg_i1 = shadow_vs_factor * (shadow_vs_period_total[1] / shadow_vs_avg_period)

        real_body_gap_down_i1_i2 = max(open_values[i - 1], close_values[i - 1]) < min(open_values[i - 2], close_values[i - 2])

        if (
            color_i3 == -1
            and color_i2 == -1
            and color_i1 == -1
            and color_i0 == -1
            and lower_shadow_i3 < shadow_vs_avg_i3
            and upper_shadow_i3 < shadow_vs_avg_i3
            and lower_shadow_i2 < shadow_vs_avg_i2
            and upper_shadow_i2 < shadow_vs_avg_i2
            and real_body_gap_down_i1_i2
            and upper_shadow_i1 > shadow_vs_avg_i1
            and high_values[i - 1] > close_values[i - 2]
            and high_values[i] > high_values[i - 1]
            and low_values[i] < low_values[i - 1]
        ):
            out[i] = 100

        for tot_idx in range(3, 0, -1):
            shadow_vs_period_total[tot_idx] += (
                (high_values[i - tot_idx] - low_values[i - tot_idx])
                - (high_values[shadow_vs_trailing_idx - tot_idx] - low_values[shadow_vs_trailing_idx - tot_idx])
            )

        i += 1
        shadow_vs_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdlconcealbabyswall', values=out))
