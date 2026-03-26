import numpy as np
import polars as pl

CDLSTALLEDPATTERN_BODY_LONG_AVG_PERIOD = 10
CDLSTALLEDPATTERN_BODY_SHORT_AVG_PERIOD = 10
CDLSTALLEDPATTERN_BODY_SHORT_PERIOD_TOTAL = 0.0
CDLSTALLEDPATTERN_NEAR_AVG_PERIOD = 5
CDLSTALLEDPATTERN_NEAR_FACTOR = 0.2
CDLSTALLEDPATTERN_SHADOW_VS_AVG_PERIOD = 10
CDLSTALLEDPATTERN_SHADOW_VS_FACTOR = 0.1
CDLSTALLEDPATTERN_SHADOW_VS_PERIOD_TOTAL = 0.0


def _cdlstalledpattern_impl(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Stalled Pattern candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdlstalledpattern'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    body_long_avg_period = CDLSTALLEDPATTERN_BODY_LONG_AVG_PERIOD
    body_short_avg_period = CDLSTALLEDPATTERN_BODY_SHORT_AVG_PERIOD
    shadow_vs_avg_period = CDLSTALLEDPATTERN_SHADOW_VS_AVG_PERIOD
    shadow_vs_factor = CDLSTALLEDPATTERN_SHADOW_VS_FACTOR
    near_avg_period = CDLSTALLEDPATTERN_NEAR_AVG_PERIOD
    near_factor = CDLSTALLEDPATTERN_NEAR_FACTOR

    lookback_total = max(body_long_avg_period, body_short_avg_period, shadow_vs_avg_period, near_avg_period) + 2

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlstalledpattern', values=out))

    body_long_period_total = np.zeros(3, dtype=float)
    near_period_total = np.zeros(3, dtype=float)
    body_short_period_total = CDLSTALLEDPATTERN_BODY_SHORT_PERIOD_TOTAL
    shadow_vs_period_total = CDLSTALLEDPATTERN_SHADOW_VS_PERIOD_TOTAL

    body_long_trailing_idx = lookback_total - body_long_avg_period
    body_short_trailing_idx = lookback_total - body_short_avg_period
    shadow_vs_trailing_idx = lookback_total - shadow_vs_avg_period
    near_trailing_idx = lookback_total - near_avg_period

    i = body_long_trailing_idx
    while i < lookback_total:
        body_long_period_total[2] += abs(close_values[i - 2] - open_values[i - 2])
        body_long_period_total[1] += abs(close_values[i - 1] - open_values[i - 1])
        i += 1

    i = body_short_trailing_idx
    while i < lookback_total:
        body_short_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = shadow_vs_trailing_idx
    while i < lookback_total:
        shadow_vs_period_total += high_values[i - 1] - low_values[i - 1]
        i += 1

    i = near_trailing_idx
    while i < lookback_total:
        near_period_total[2] += high_values[i - 2] - low_values[i - 2]
        near_period_total[1] += high_values[i - 1] - low_values[i - 1]
        i += 1

    i = lookback_total
    while i < n:
        real_body_i2 = abs(close_values[i - 2] - open_values[i - 2])
        real_body_i1 = abs(close_values[i - 1] - open_values[i - 1])
        real_body_i0 = abs(close_values[i] - open_values[i])
        upper_shadow_i1 = high_values[i - 1] - max(open_values[i - 1], close_values[i - 1])

        body_long_avg_i2 = body_long_period_total[2] / body_long_avg_period
        body_long_avg_i1 = body_long_period_total[1] / body_long_avg_period
        body_short_avg_i0 = body_short_period_total / body_short_avg_period
        shadow_vs_avg_i1 = shadow_vs_factor * (shadow_vs_period_total / shadow_vs_avg_period)
        near_avg_i2 = near_factor * (near_period_total[2] / near_avg_period)
        near_avg_i1 = near_factor * (near_period_total[1] / near_avg_period)

        if (
            close_values[i - 2] >= open_values[i - 2]
            and close_values[i - 1] >= open_values[i - 1]
            and close_values[i] >= open_values[i]
            and close_values[i] > close_values[i - 1]
            and close_values[i - 1] > close_values[i - 2]
            and real_body_i2 > body_long_avg_i2
            and real_body_i1 > body_long_avg_i1
            and upper_shadow_i1 < shadow_vs_avg_i1
            and open_values[i - 1] > open_values[i - 2]
            and open_values[i - 1] <= close_values[i - 2] + near_avg_i2
            and real_body_i0 < body_short_avg_i0
            and open_values[i] >= close_values[i - 1] - real_body_i0 - near_avg_i1
        ):
            out[i] = -100

        for tot_idx in (2, 1):
            body_long_period_total[tot_idx] += (
                abs(close_values[i - tot_idx] - open_values[i - tot_idx])
                - abs(close_values[body_long_trailing_idx - tot_idx] - open_values[body_long_trailing_idx - tot_idx])
            )
            near_period_total[tot_idx] += (
                (high_values[i - tot_idx] - low_values[i - tot_idx])
                - (high_values[near_trailing_idx - tot_idx] - low_values[near_trailing_idx - tot_idx])
            )

        body_short_period_total += (
            real_body_i0
            - abs(close_values[body_short_trailing_idx] - open_values[body_short_trailing_idx])
        )
        shadow_vs_period_total += (
            (high_values[i - 1] - low_values[i - 1])
            - (high_values[shadow_vs_trailing_idx - 1] - low_values[shadow_vs_trailing_idx - 1])
        )

        i += 1
        body_long_trailing_idx += 1
        body_short_trailing_idx += 1
        shadow_vs_trailing_idx += 1
        near_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdlstalledpattern', values=out))


def cdlstalledpattern(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    out_col = 'cdlstalledpattern'
    input_cols = [open_col, high_col, low_col, close_col]
    return data.with_columns(
        pl.struct(input_cols).map_batches(
            lambda s: _cdlstalledpattern_impl(
                pl.DataFrame({col: s.struct.field(col) for col in input_cols}),
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            ).get_column(out_col),
            return_dtype=pl.Int32,
        ).alias(out_col)
    )
