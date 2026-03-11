import numpy as np
import polars as pl

CDLRICKSHAWMAN_BODY_DOJI_AVG_PERIOD = 10
CDLRICKSHAWMAN_BODY_DOJI_FACTOR = 0.1
CDLRICKSHAWMAN_BODY_DOJI_PERIOD_TOTAL = 0.0
CDLRICKSHAWMAN_NEAR_AVG_PERIOD = 5
CDLRICKSHAWMAN_NEAR_FACTOR = 0.2
CDLRICKSHAWMAN_NEAR_PERIOD_TOTAL = 0.0
CDLRICKSHAWMAN_SHADOW_LONG_AVG_PERIOD = 0
CDLRICKSHAWMAN_SHADOW_LONG_FACTOR = 1.0
CDLRICKSHAWMAN_SHADOW_LONG_PERIOD_TOTAL = 0.0


def _cdlrickshawman_impl(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Rickshaw Man candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdlrickshawman'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    body_doji_avg_period = CDLRICKSHAWMAN_BODY_DOJI_AVG_PERIOD
    body_doji_factor = CDLRICKSHAWMAN_BODY_DOJI_FACTOR
    shadow_long_avg_period = CDLRICKSHAWMAN_SHADOW_LONG_AVG_PERIOD
    shadow_long_factor = CDLRICKSHAWMAN_SHADOW_LONG_FACTOR
    near_avg_period = CDLRICKSHAWMAN_NEAR_AVG_PERIOD
    near_factor = CDLRICKSHAWMAN_NEAR_FACTOR
    lookback_total = max(body_doji_avg_period, shadow_long_avg_period, near_avg_period)

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlrickshawman', values=out))

    body_doji_period_total = CDLRICKSHAWMAN_BODY_DOJI_PERIOD_TOTAL
    shadow_long_period_total = CDLRICKSHAWMAN_SHADOW_LONG_PERIOD_TOTAL
    near_period_total = CDLRICKSHAWMAN_NEAR_PERIOD_TOTAL
    body_doji_trailing_idx = lookback_total - body_doji_avg_period
    shadow_long_trailing_idx = lookback_total - shadow_long_avg_period
    near_trailing_idx = lookback_total - near_avg_period

    i = body_doji_trailing_idx
    while i < lookback_total:
        body_doji_period_total += high_values[i] - low_values[i]
        i += 1

    i = shadow_long_trailing_idx
    while i < lookback_total:
        shadow_long_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = near_trailing_idx
    while i < lookback_total:
        near_period_total += high_values[i] - low_values[i]
        i += 1

    i = lookback_total
    while i < n:
        real_body = abs(close_values[i] - open_values[i])
        lower_shadow = min(open_values[i], close_values[i]) - low_values[i]
        upper_shadow = high_values[i] - max(open_values[i], close_values[i])
        high_low_range = high_values[i] - low_values[i]
        midpoint = low_values[i] + (high_low_range / 2.0)
        body_low = min(open_values[i], close_values[i])
        body_high = max(open_values[i], close_values[i])

        body_doji_avg = body_doji_factor * (body_doji_period_total / body_doji_avg_period)
        near_avg = near_factor * (near_period_total / near_avg_period)
        if shadow_long_avg_period == 0:
            shadow_long_avg = shadow_long_factor * real_body
        else:
            shadow_long_avg = shadow_long_factor * (shadow_long_period_total / shadow_long_avg_period)

        if (
            real_body <= body_doji_avg
            and lower_shadow > shadow_long_avg
            and upper_shadow > shadow_long_avg
            and body_low <= midpoint + near_avg
            and body_high >= midpoint - near_avg
        ):
            out[i] = 100

        body_doji_period_total += (
            (high_values[i] - low_values[i])
            - (high_values[body_doji_trailing_idx] - low_values[body_doji_trailing_idx])
        )
        shadow_long_period_total += (
            abs(close_values[i] - open_values[i])
            - abs(close_values[shadow_long_trailing_idx] - open_values[shadow_long_trailing_idx])
        )
        near_period_total += (
            (high_values[i] - low_values[i])
            - (high_values[near_trailing_idx] - low_values[near_trailing_idx])
        )

        i += 1
        body_doji_trailing_idx += 1
        shadow_long_trailing_idx += 1
        near_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdlrickshawman', values=out))


def cdlrickshawman(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    out_col = 'cdlrickshawman'
    input_cols = [open_col, high_col, low_col, close_col]
    return data.with_columns(
        pl.struct(input_cols).map_batches(
            lambda s: _cdlrickshawman_impl(
                pl.DataFrame({col: s.struct.field(col) for col in input_cols}),
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            ).get_column(out_col),
            return_dtype=pl.Int32,
        ).alias(out_col)
    )
