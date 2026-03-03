import numpy as np
import polars as pl


def cdlladderbottom(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Ladder Bottom candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdlladderbottom'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)

    # TA-Lib default candle setting for ShadowVeryShort:
    # rangeType=HighLow, avgPeriod=10, factor=0.1
    shadow_vs_avg_period = 10
    shadow_vs_factor = 0.1
    lookback_total = shadow_vs_avg_period + 4

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlladderbottom', values=out))

    shadow_vs_period_total = 0.0
    shadow_vs_trailing_idx = lookback_total - shadow_vs_avg_period

    i = shadow_vs_trailing_idx
    while i < lookback_total:
        shadow_vs_period_total += high_values[i - 1] - low_values[i - 1]
        i += 1

    i = lookback_total
    while i < n:
        color_i4 = 1 if close_values[i - 4] >= open_values[i - 4] else -1
        color_i3 = 1 if close_values[i - 3] >= open_values[i - 3] else -1
        color_i2 = 1 if close_values[i - 2] >= open_values[i - 2] else -1
        color_i1 = 1 if close_values[i - 1] >= open_values[i - 1] else -1
        color_i0 = 1 if close_values[i] >= open_values[i] else -1

        upper_shadow_i1 = high_values[i - 1] - max(open_values[i - 1], close_values[i - 1])
        shadow_vs_avg_i1 = shadow_vs_factor * (shadow_vs_period_total / shadow_vs_avg_period)

        if (
            color_i4 == -1
            and color_i3 == -1
            and color_i2 == -1
            and open_values[i - 4] > open_values[i - 3]
            and open_values[i - 3] > open_values[i - 2]
            and close_values[i - 4] > close_values[i - 3]
            and close_values[i - 3] > close_values[i - 2]
            and color_i1 == -1
            and upper_shadow_i1 > shadow_vs_avg_i1
            and color_i0 == 1
            and open_values[i] > open_values[i - 1]
            and close_values[i] > high_values[i - 1]
        ):
            out[i] = 100

        shadow_vs_period_total += (
            (high_values[i - 1] - low_values[i - 1])
            - (high_values[shadow_vs_trailing_idx - 1] - low_values[shadow_vs_trailing_idx - 1])
        )
        i += 1
        shadow_vs_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdlladderbottom', values=out))
