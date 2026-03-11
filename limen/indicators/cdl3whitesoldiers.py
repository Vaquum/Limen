import numpy as np
import polars as pl

CDL3WHITESOLDIERS_BODY_SHORT_AVG_PERIOD = 10
CDL3WHITESOLDIERS_BODY_SHORT_PERIOD_TOTAL = 0.0
CDL3WHITESOLDIERS_FAR_AVG_PERIOD = 5
CDL3WHITESOLDIERS_FAR_FACTOR = 0.6
CDL3WHITESOLDIERS_NEAR_AVG_PERIOD = 5
CDL3WHITESOLDIERS_NEAR_FACTOR = 0.2
CDL3WHITESOLDIERS_SHADOW_VS_AVG_PERIOD = 10
CDL3WHITESOLDIERS_SHADOW_VS_FACTOR = 0.1


def _cdl3whitesoldiers_impl(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Three Advancing White Soldiers candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdl3whitesoldiers'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    shadow_vs_avg_period = CDL3WHITESOLDIERS_SHADOW_VS_AVG_PERIOD
    shadow_vs_factor = CDL3WHITESOLDIERS_SHADOW_VS_FACTOR
    body_short_avg_period = CDL3WHITESOLDIERS_BODY_SHORT_AVG_PERIOD
    near_avg_period = CDL3WHITESOLDIERS_NEAR_AVG_PERIOD
    near_factor = CDL3WHITESOLDIERS_NEAR_FACTOR
    far_avg_period = CDL3WHITESOLDIERS_FAR_AVG_PERIOD
    far_factor = CDL3WHITESOLDIERS_FAR_FACTOR

    lookback_total = max(shadow_vs_avg_period, body_short_avg_period, far_avg_period, near_avg_period) + 2

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdl3whitesoldiers', values=out))

    shadow_vs_period_total = np.zeros(3, dtype=float)
    near_period_total = np.zeros(3, dtype=float)
    far_period_total = np.zeros(3, dtype=float)
    body_short_period_total = CDL3WHITESOLDIERS_BODY_SHORT_PERIOD_TOTAL

    shadow_vs_trailing_idx = lookback_total - shadow_vs_avg_period
    near_trailing_idx = lookback_total - near_avg_period
    far_trailing_idx = lookback_total - far_avg_period
    body_short_trailing_idx = lookback_total - body_short_avg_period

    i = shadow_vs_trailing_idx
    while i < lookback_total:
        shadow_vs_period_total[2] += high_values[i - 2] - low_values[i - 2]
        shadow_vs_period_total[1] += high_values[i - 1] - low_values[i - 1]
        shadow_vs_period_total[0] += high_values[i] - low_values[i]
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

    i = body_short_trailing_idx
    while i < lookback_total:
        body_short_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = lookback_total
    while i < n:
        upper_shadow_i2 = high_values[i - 2] - max(open_values[i - 2], close_values[i - 2])
        upper_shadow_i1 = high_values[i - 1] - max(open_values[i - 1], close_values[i - 1])
        upper_shadow_i0 = high_values[i] - max(open_values[i], close_values[i])

        real_body_i2 = abs(close_values[i - 2] - open_values[i - 2])
        real_body_i1 = abs(close_values[i - 1] - open_values[i - 1])
        real_body_i0 = abs(close_values[i] - open_values[i])

        shadow_vs_avg_i2 = shadow_vs_factor * (shadow_vs_period_total[2] / shadow_vs_avg_period)
        shadow_vs_avg_i1 = shadow_vs_factor * (shadow_vs_period_total[1] / shadow_vs_avg_period)
        shadow_vs_avg_i0 = shadow_vs_factor * (shadow_vs_period_total[0] / shadow_vs_avg_period)
        near_avg_i2 = near_factor * (near_period_total[2] / near_avg_period)
        near_avg_i1 = near_factor * (near_period_total[1] / near_avg_period)
        far_avg_i2 = far_factor * (far_period_total[2] / far_avg_period)
        far_avg_i1 = far_factor * (far_period_total[1] / far_avg_period)
        body_short_avg_i0 = body_short_period_total / body_short_avg_period

        if (
            close_values[i - 2] >= open_values[i - 2]
            and upper_shadow_i2 < shadow_vs_avg_i2
            and close_values[i - 1] >= open_values[i - 1]
            and upper_shadow_i1 < shadow_vs_avg_i1
            and close_values[i] >= open_values[i]
            and upper_shadow_i0 < shadow_vs_avg_i0
            and close_values[i] > close_values[i - 1]
            and close_values[i - 1] > close_values[i - 2]
            and open_values[i - 1] > open_values[i - 2]
            and open_values[i - 1] <= close_values[i - 2] + near_avg_i2
            and open_values[i] > open_values[i - 1]
            and open_values[i] <= close_values[i - 1] + near_avg_i1
            and real_body_i1 > real_body_i2 - far_avg_i2
            and real_body_i0 > real_body_i1 - far_avg_i1
            and real_body_i0 > body_short_avg_i0
        ):
            out[i] = 100

        for tot_idx in range(2, -1, -1):
            shadow_vs_period_total[tot_idx] += (
                (high_values[i - tot_idx] - low_values[i - tot_idx])
                - (high_values[shadow_vs_trailing_idx - tot_idx] - low_values[shadow_vs_trailing_idx - tot_idx])
            )

        for tot_idx in range(2, 0, -1):
            far_period_total[tot_idx] += (
                (high_values[i - tot_idx] - low_values[i - tot_idx])
                - (high_values[far_trailing_idx - tot_idx] - low_values[far_trailing_idx - tot_idx])
            )
            near_period_total[tot_idx] += (
                (high_values[i - tot_idx] - low_values[i - tot_idx])
                - (high_values[near_trailing_idx - tot_idx] - low_values[near_trailing_idx - tot_idx])
            )

        body_short_period_total += (
            abs(close_values[i] - open_values[i])
            - abs(close_values[body_short_trailing_idx] - open_values[body_short_trailing_idx])
        )

        i += 1
        shadow_vs_trailing_idx += 1
        near_trailing_idx += 1
        far_trailing_idx += 1
        body_short_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdl3whitesoldiers', values=out))


def cdl3whitesoldiers(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    out_col = 'cdl3whitesoldiers'
    input_cols = [open_col, high_col, low_col, close_col]
    return data.with_columns(
        pl.struct(input_cols).map_batches(
            lambda s: _cdl3whitesoldiers_impl(
                pl.DataFrame({col: s.struct.field(col) for col in input_cols}),
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            ).get_column(out_col),
            return_dtype=pl.Int32,
        ).alias(out_col)
    )
