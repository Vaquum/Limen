import numpy as np
import polars as pl


def cdl3starsinsouth(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Three Stars In The South candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdl3starsinsouth'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)

    # TA-Lib default candle settings:
    # BodyLong: rangeType=RealBody, avgPeriod=10, factor=1.0
    # BodyShort: rangeType=RealBody, avgPeriod=10, factor=1.0
    # ShadowLong: rangeType=RealBody, avgPeriod=0, factor=1.0
    # ShadowVeryShort: rangeType=HighLow, avgPeriod=10, factor=0.1
    body_long_avg_period = 10
    body_short_avg_period = 10
    shadow_long_avg_period = 0
    shadow_vs_avg_period = 10
    shadow_vs_factor = 0.1

    lookback_total = max(
        max(shadow_vs_avg_period, shadow_long_avg_period),
        max(body_long_avg_period, body_short_avg_period),
    ) + 2

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdl3starsinsouth', values=out))

    body_long_period_total = 0.0
    body_short_period_total = 0.0
    shadow_long_period_total = 0.0
    shadow_vs_period_total = np.zeros(2, dtype=float)

    body_long_trailing_idx = lookback_total - body_long_avg_period
    body_short_trailing_idx = lookback_total - body_short_avg_period
    shadow_long_trailing_idx = lookback_total - shadow_long_avg_period
    shadow_vs_trailing_idx = lookback_total - shadow_vs_avg_period

    i = body_long_trailing_idx
    while i < lookback_total:
        body_long_period_total += abs(close_values[i - 2] - open_values[i - 2])
        i += 1

    i = shadow_long_trailing_idx
    while i < lookback_total:
        shadow_long_period_total += abs(close_values[i - 2] - open_values[i - 2])
        i += 1

    i = shadow_vs_trailing_idx
    while i < lookback_total:
        shadow_vs_period_total[1] += high_values[i - 1] - low_values[i - 1]
        shadow_vs_period_total[0] += high_values[i] - low_values[i]
        i += 1

    i = body_short_trailing_idx
    while i < lookback_total:
        body_short_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = lookback_total
    while i < n:
        real_body_i2 = abs(close_values[i - 2] - open_values[i - 2])
        real_body_i1 = abs(close_values[i - 1] - open_values[i - 1])
        real_body_i0 = abs(close_values[i] - open_values[i])

        lower_shadow_i2 = min(open_values[i - 2], close_values[i - 2]) - low_values[i - 2]
        lower_shadow_i1 = min(open_values[i - 1], close_values[i - 1]) - low_values[i - 1]
        lower_shadow_i0 = min(open_values[i], close_values[i]) - low_values[i]
        upper_shadow_i0 = high_values[i] - max(open_values[i], close_values[i])

        body_long_avg_i2 = body_long_period_total / body_long_avg_period
        body_short_avg_i0 = body_short_period_total / body_short_avg_period
        shadow_vs_avg_i1 = shadow_vs_factor * (shadow_vs_period_total[1] / shadow_vs_avg_period)
        shadow_vs_avg_i0 = shadow_vs_factor * (shadow_vs_period_total[0] / shadow_vs_avg_period)

        if shadow_long_avg_period == 0:
            shadow_long_avg_i2 = real_body_i2
        else:
            shadow_long_avg_i2 = shadow_long_period_total / shadow_long_avg_period

        if (
            close_values[i - 2] < open_values[i - 2]
            and close_values[i - 1] < open_values[i - 1]
            and close_values[i] < open_values[i]
            and real_body_i2 > body_long_avg_i2
            and lower_shadow_i2 > shadow_long_avg_i2
            and real_body_i1 < real_body_i2
            and open_values[i - 1] > close_values[i - 2]
            and open_values[i - 1] <= high_values[i - 2]
            and low_values[i - 1] < close_values[i - 2]
            and low_values[i - 1] >= low_values[i - 2]
            and lower_shadow_i1 > shadow_vs_avg_i1
            and real_body_i0 < body_short_avg_i0
            and lower_shadow_i0 < shadow_vs_avg_i0
            and upper_shadow_i0 < shadow_vs_avg_i0
            and low_values[i] > low_values[i - 1]
            and high_values[i] < high_values[i - 1]
        ):
            out[i] = 100

        shadow_long_period_total += (
            abs(close_values[i - 2] - open_values[i - 2])
            - abs(close_values[shadow_long_trailing_idx - 2] - open_values[shadow_long_trailing_idx - 2])
        )
        body_long_period_total += (
            abs(close_values[i - 2] - open_values[i - 2])
            - abs(close_values[body_long_trailing_idx - 2] - open_values[body_long_trailing_idx - 2])
        )
        for tot_idx in range(1, -1, -1):
            shadow_vs_period_total[tot_idx] += (
                (high_values[i - tot_idx] - low_values[i - tot_idx])
                - (high_values[shadow_vs_trailing_idx - tot_idx] - low_values[shadow_vs_trailing_idx - tot_idx])
            )
        body_short_period_total += (
            abs(close_values[i] - open_values[i])
            - abs(close_values[body_short_trailing_idx] - open_values[body_short_trailing_idx])
        )

        i += 1
        body_long_trailing_idx += 1
        body_short_trailing_idx += 1
        shadow_long_trailing_idx += 1
        shadow_vs_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdl3starsinsouth', values=out))
