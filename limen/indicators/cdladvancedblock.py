import numpy as np
import polars as pl


def cdladvancedblock(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Advance Block candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdladvancedblock'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)

    # TA-Lib default candle settings:
    # BodyLong: rangeType=RealBody, avgPeriod=10, factor=1.0
    # ShadowShort: rangeType=Shadows, avgPeriod=10, factor=1.0
    # ShadowLong: rangeType=RealBody, avgPeriod=0, factor=1.0
    # Near: rangeType=HighLow, avgPeriod=5, factor=0.2
    # Far: rangeType=HighLow, avgPeriod=5, factor=0.6
    body_long_avg_period = 10
    shadow_short_avg_period = 10
    shadow_long_avg_period = 0
    near_avg_period = 5
    near_factor = 0.2
    far_avg_period = 5
    far_factor = 0.6

    lookback_total = max(
        max(max(shadow_long_avg_period, shadow_short_avg_period), max(far_avg_period, near_avg_period)),
        body_long_avg_period,
    ) + 2

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdladvancedblock', values=out))

    shadow_short_period_total = np.zeros(3, dtype=float)
    shadow_long_period_total = np.zeros(2, dtype=float)
    near_period_total = np.zeros(3, dtype=float)
    far_period_total = np.zeros(3, dtype=float)
    body_long_period_total = 0.0

    body_long_trailing_idx = lookback_total - body_long_avg_period
    shadow_short_trailing_idx = lookback_total - shadow_short_avg_period
    shadow_long_trailing_idx = lookback_total - shadow_long_avg_period
    near_trailing_idx = lookback_total - near_avg_period
    far_trailing_idx = lookback_total - far_avg_period

    i = shadow_short_trailing_idx
    while i < lookback_total:
        upper_shadow_i2 = high_values[i - 2] - max(open_values[i - 2], close_values[i - 2])
        lower_shadow_i2 = min(open_values[i - 2], close_values[i - 2]) - low_values[i - 2]
        upper_shadow_i1 = high_values[i - 1] - max(open_values[i - 1], close_values[i - 1])
        lower_shadow_i1 = min(open_values[i - 1], close_values[i - 1]) - low_values[i - 1]
        upper_shadow_i0 = high_values[i] - max(open_values[i], close_values[i])
        lower_shadow_i0 = min(open_values[i], close_values[i]) - low_values[i]
        shadow_short_period_total[2] += upper_shadow_i2 + lower_shadow_i2
        shadow_short_period_total[1] += upper_shadow_i1 + lower_shadow_i1
        shadow_short_period_total[0] += upper_shadow_i0 + lower_shadow_i0
        i += 1

    i = shadow_long_trailing_idx
    while i < lookback_total:
        shadow_long_period_total[1] += abs(close_values[i - 1] - open_values[i - 1])
        shadow_long_period_total[0] += abs(close_values[i] - open_values[i])
        i += 1

    i = near_trailing_idx
    while i < lookback_total:
        near_period_total[2] += high_values[i - 2] - low_values[i - 2]
        near_period_total[1] += high_values[i - 1] - low_values[i - 1]
        i += 1

    i = far_trailing_idx
    while i < lookback_total:
        far_period_total[2] += high_values[i - 2] - low_values[i - 2]
        far_period_total[1] += high_values[i - 1] - low_values[i - 1]
        i += 1

    i = body_long_trailing_idx
    while i < lookback_total:
        body_long_period_total += abs(close_values[i - 2] - open_values[i - 2])
        i += 1

    i = lookback_total
    while i < n:
        real_body_i2 = abs(close_values[i - 2] - open_values[i - 2])
        real_body_i1 = abs(close_values[i - 1] - open_values[i - 1])
        real_body_i0 = abs(close_values[i] - open_values[i])

        upper_shadow_i2 = high_values[i - 2] - max(open_values[i - 2], close_values[i - 2])
        lower_shadow_i2 = min(open_values[i - 2], close_values[i - 2]) - low_values[i - 2]
        upper_shadow_i1 = high_values[i - 1] - max(open_values[i - 1], close_values[i - 1])
        lower_shadow_i1 = min(open_values[i - 1], close_values[i - 1]) - low_values[i - 1]
        upper_shadow_i0 = high_values[i] - max(open_values[i], close_values[i])
        lower_shadow_i0 = min(open_values[i], close_values[i]) - low_values[i]

        # ShadowShort uses Shadows range type, TA_CANDLEAVERAGE divides by 2.0.
        shadow_short_avg_i2 = (shadow_short_period_total[2] / shadow_short_avg_period) / 2.0
        shadow_short_avg_i1 = (shadow_short_period_total[1] / shadow_short_avg_period) / 2.0
        shadow_short_avg_i0 = (shadow_short_period_total[0] / shadow_short_avg_period) / 2.0
        near_avg_i2 = near_factor * (near_period_total[2] / near_avg_period)
        near_avg_i1 = near_factor * (near_period_total[1] / near_avg_period)
        far_avg_i2 = far_factor * (far_period_total[2] / far_avg_period)
        far_avg_i1 = far_factor * (far_period_total[1] / far_avg_period)
        body_long_avg_i2 = body_long_period_total / body_long_avg_period

        if shadow_long_avg_period == 0:
            shadow_long_avg_i0 = real_body_i0
        else:
            shadow_long_avg_i0 = shadow_long_period_total[0] / shadow_long_avg_period

        first_white = close_values[i - 2] >= open_values[i - 2]
        second_white = close_values[i - 1] >= open_values[i - 1]
        third_white = close_values[i] >= open_values[i]

        if (
            first_white
            and second_white
            and third_white
            and close_values[i] > close_values[i - 1]
            and close_values[i - 1] > close_values[i - 2]
            and open_values[i - 1] > open_values[i - 2]
            and open_values[i - 1] <= close_values[i - 2] + near_avg_i2
            and open_values[i] > open_values[i - 1]
            and open_values[i] <= close_values[i - 1] + near_avg_i1
            and real_body_i2 > body_long_avg_i2
            and upper_shadow_i2 < shadow_short_avg_i2
            and (
                (
                    real_body_i1 < real_body_i2 - far_avg_i2
                    and real_body_i0 < real_body_i1 + near_avg_i1
                )
                or (real_body_i0 < real_body_i1 - far_avg_i1)
                or (
                    real_body_i0 < real_body_i1
                    and real_body_i1 < real_body_i2
                    and (upper_shadow_i0 > shadow_short_avg_i0 or upper_shadow_i1 > shadow_short_avg_i1)
                )
                or (
                    real_body_i0 < real_body_i1
                    and upper_shadow_i0 > shadow_long_avg_i0
                )
            )
        ):
            out[i] = -100

        shadow_short_period_total[2] += (
            (upper_shadow_i2 + lower_shadow_i2)
            - (
                (high_values[shadow_short_trailing_idx - 2] - max(open_values[shadow_short_trailing_idx - 2], close_values[shadow_short_trailing_idx - 2]))
                + (min(open_values[shadow_short_trailing_idx - 2], close_values[shadow_short_trailing_idx - 2]) - low_values[shadow_short_trailing_idx - 2])
            )
        )
        shadow_short_period_total[1] += (
            (upper_shadow_i1 + lower_shadow_i1)
            - (
                (high_values[shadow_short_trailing_idx - 1] - max(open_values[shadow_short_trailing_idx - 1], close_values[shadow_short_trailing_idx - 1]))
                + (min(open_values[shadow_short_trailing_idx - 1], close_values[shadow_short_trailing_idx - 1]) - low_values[shadow_short_trailing_idx - 1])
            )
        )
        shadow_short_period_total[0] += (
            (upper_shadow_i0 + lower_shadow_i0)
            - (
                (high_values[shadow_short_trailing_idx] - max(open_values[shadow_short_trailing_idx], close_values[shadow_short_trailing_idx]))
                + (min(open_values[shadow_short_trailing_idx], close_values[shadow_short_trailing_idx]) - low_values[shadow_short_trailing_idx])
            )
        )

        shadow_long_period_total[1] += (
            real_body_i1
            - abs(close_values[shadow_long_trailing_idx - 1] - open_values[shadow_long_trailing_idx - 1])
        )
        shadow_long_period_total[0] += (
            real_body_i0
            - abs(close_values[shadow_long_trailing_idx] - open_values[shadow_long_trailing_idx])
        )

        far_period_total[2] += (
            (high_values[i - 2] - low_values[i - 2])
            - (high_values[far_trailing_idx - 2] - low_values[far_trailing_idx - 2])
        )
        far_period_total[1] += (
            (high_values[i - 1] - low_values[i - 1])
            - (high_values[far_trailing_idx - 1] - low_values[far_trailing_idx - 1])
        )

        near_period_total[2] += (
            (high_values[i - 2] - low_values[i - 2])
            - (high_values[near_trailing_idx - 2] - low_values[near_trailing_idx - 2])
        )
        near_period_total[1] += (
            (high_values[i - 1] - low_values[i - 1])
            - (high_values[near_trailing_idx - 1] - low_values[near_trailing_idx - 1])
        )

        body_long_period_total += (
            real_body_i2
            - abs(close_values[body_long_trailing_idx - 2] - open_values[body_long_trailing_idx - 2])
        )

        i += 1
        shadow_short_trailing_idx += 1
        shadow_long_trailing_idx += 1
        near_trailing_idx += 1
        far_trailing_idx += 1
        body_long_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdladvancedblock', values=out))
