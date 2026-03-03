import numpy as np
import polars as pl

CDL3BLACKCROWS_SHADOW_VS_AVG_PERIOD = 10
CDL3BLACKCROWS_SHADOW_VS_FACTOR = 0.1


def cdl3blackcrows(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Three Black Crows candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdl3blackcrows'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    shadow_vs_avg_period = CDL3BLACKCROWS_SHADOW_VS_AVG_PERIOD
    shadow_vs_factor = CDL3BLACKCROWS_SHADOW_VS_FACTOR
    lookback_total = shadow_vs_avg_period + 3

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdl3blackcrows', values=out))

    shadow_vs_period_total = np.zeros(3, dtype=float)
    shadow_vs_trailing_idx = lookback_total - shadow_vs_avg_period

    i = shadow_vs_trailing_idx
    while i < lookback_total:
        shadow_vs_period_total[2] += high_values[i - 2] - low_values[i - 2]
        shadow_vs_period_total[1] += high_values[i - 1] - low_values[i - 1]
        shadow_vs_period_total[0] += high_values[i] - low_values[i]
        i += 1

    i = lookback_total
    while i < n:
        lower_shadow_i2 = min(open_values[i - 2], close_values[i - 2]) - low_values[i - 2]
        lower_shadow_i1 = min(open_values[i - 1], close_values[i - 1]) - low_values[i - 1]
        lower_shadow_i0 = min(open_values[i], close_values[i]) - low_values[i]

        shadow_vs_avg_i2 = shadow_vs_factor * (shadow_vs_period_total[2] / shadow_vs_avg_period)
        shadow_vs_avg_i1 = shadow_vs_factor * (shadow_vs_period_total[1] / shadow_vs_avg_period)
        shadow_vs_avg_i0 = shadow_vs_factor * (shadow_vs_period_total[0] / shadow_vs_avg_period)

        if (
            close_values[i - 3] >= open_values[i - 3]
            and close_values[i - 2] < open_values[i - 2]
            and lower_shadow_i2 < shadow_vs_avg_i2
            and close_values[i - 1] < open_values[i - 1]
            and lower_shadow_i1 < shadow_vs_avg_i1
            and close_values[i] < open_values[i]
            and lower_shadow_i0 < shadow_vs_avg_i0
            and open_values[i - 1] < open_values[i - 2]
            and open_values[i - 1] > close_values[i - 2]
            and open_values[i] < open_values[i - 1]
            and open_values[i] > close_values[i - 1]
            and high_values[i - 3] > close_values[i - 2]
            and close_values[i - 2] > close_values[i - 1]
            and close_values[i - 1] > close_values[i]
        ):
            out[i] = -100

        for tot_idx in range(2, -1, -1):
            shadow_vs_period_total[tot_idx] += (
                (high_values[i - tot_idx] - low_values[i - tot_idx])
                - (high_values[shadow_vs_trailing_idx - tot_idx] - low_values[shadow_vs_trailing_idx - tot_idx])
            )

        i += 1
        shadow_vs_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdl3blackcrows', values=out))
