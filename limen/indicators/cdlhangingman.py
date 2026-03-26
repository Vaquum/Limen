import numpy as np
import polars as pl

CDLHANGINGMAN_BODY_PERIOD_TOTAL = 0.0
CDLHANGINGMAN_BODY_SHORT_AVG_PERIOD = 10
CDLHANGINGMAN_NEAR_AVG_PERIOD = 5
CDLHANGINGMAN_NEAR_FACTOR = 0.2
CDLHANGINGMAN_NEAR_PERIOD_TOTAL = 0.0
CDLHANGINGMAN_SHADOW_LONG_AVG_PERIOD = 0
CDLHANGINGMAN_SHADOW_LONG_FACTOR = 1.0
CDLHANGINGMAN_SHADOW_LONG_PERIOD_TOTAL = 0.0
CDLHANGINGMAN_SHADOW_VS_AVG_PERIOD = 10
CDLHANGINGMAN_SHADOW_VS_FACTOR = 0.1
CDLHANGINGMAN_SHADOW_VS_PERIOD_TOTAL = 0.0


def _cdlhangingman_impl(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Hanging Man candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdlhangingman'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    body_short_avg_period = CDLHANGINGMAN_BODY_SHORT_AVG_PERIOD
    shadow_long_avg_period = CDLHANGINGMAN_SHADOW_LONG_AVG_PERIOD
    shadow_long_factor = CDLHANGINGMAN_SHADOW_LONG_FACTOR
    shadow_vs_avg_period = CDLHANGINGMAN_SHADOW_VS_AVG_PERIOD
    shadow_vs_factor = CDLHANGINGMAN_SHADOW_VS_FACTOR
    near_avg_period = CDLHANGINGMAN_NEAR_AVG_PERIOD
    near_factor = CDLHANGINGMAN_NEAR_FACTOR

    lookback_total = max(body_short_avg_period, shadow_long_avg_period, shadow_vs_avg_period, near_avg_period) + 1

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlhangingman', values=out))

    body_period_total = CDLHANGINGMAN_BODY_PERIOD_TOTAL
    shadow_long_period_total = CDLHANGINGMAN_SHADOW_LONG_PERIOD_TOTAL
    shadow_vs_period_total = CDLHANGINGMAN_SHADOW_VS_PERIOD_TOTAL
    near_period_total = CDLHANGINGMAN_NEAR_PERIOD_TOTAL

    body_trailing_idx = lookback_total - body_short_avg_period
    shadow_long_trailing_idx = lookback_total - shadow_long_avg_period
    shadow_vs_trailing_idx = lookback_total - shadow_vs_avg_period
    near_trailing_idx = lookback_total - 1 - near_avg_period

    i = body_trailing_idx
    while i < lookback_total:
        body_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = shadow_long_trailing_idx
    while i < lookback_total:
        shadow_long_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = shadow_vs_trailing_idx
    while i < lookback_total:
        shadow_vs_period_total += high_values[i] - low_values[i]
        i += 1

    i = near_trailing_idx
    while i < lookback_total - 1:
        near_period_total += high_values[i] - low_values[i]
        i += 1

    i = lookback_total
    while i < n:
        real_body = abs(close_values[i] - open_values[i])
        lower_shadow = min(open_values[i], close_values[i]) - low_values[i]
        upper_shadow = high_values[i] - max(open_values[i], close_values[i])

        body_avg = body_period_total / body_short_avg_period
        shadow_vs_avg = shadow_vs_factor * (shadow_vs_period_total / shadow_vs_avg_period)
        near_avg = near_factor * (near_period_total / near_avg_period)

        if shadow_long_avg_period == 0:
            shadow_long_avg = shadow_long_factor * real_body
        else:
            shadow_long_avg = shadow_long_factor * (shadow_long_period_total / shadow_long_avg_period)

        if (
            real_body < body_avg
            and lower_shadow > shadow_long_avg
            and upper_shadow < shadow_vs_avg
            and min(open_values[i], close_values[i]) >= high_values[i - 1] - near_avg
        ):
            out[i] = -100

        body_period_total += (
            abs(close_values[i] - open_values[i])
            - abs(close_values[body_trailing_idx] - open_values[body_trailing_idx])
        )
        shadow_long_period_total += (
            abs(close_values[i] - open_values[i])
            - abs(close_values[shadow_long_trailing_idx] - open_values[shadow_long_trailing_idx])
        )
        shadow_vs_period_total += (
            (high_values[i] - low_values[i])
            - (high_values[shadow_vs_trailing_idx] - low_values[shadow_vs_trailing_idx])
        )
        near_period_total += (
            (high_values[i - 1] - low_values[i - 1])
            - (high_values[near_trailing_idx] - low_values[near_trailing_idx])
        )

        i += 1
        body_trailing_idx += 1
        shadow_long_trailing_idx += 1
        shadow_vs_trailing_idx += 1
        near_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdlhangingman', values=out))


def cdlhangingman(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    out_col = 'cdlhangingman'
    input_cols = [open_col, high_col, low_col, close_col]
    return data.with_columns(
        pl.struct(input_cols).map_batches(
            lambda s: _cdlhangingman_impl(
                pl.DataFrame({col: s.struct.field(col) for col in input_cols}),
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            ).get_column(out_col),
            return_dtype=pl.Int32,
        ).alias(out_col)
    )
