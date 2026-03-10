import numpy as np
import polars as pl

CDLLONGLINE_BODY_LONG_AVG_PERIOD = 10
CDLLONGLINE_BODY_PERIOD_TOTAL = 0.0
CDLLONGLINE_SHADOW_PERIOD_TOTAL = 0.0
CDLLONGLINE_SHADOW_SHORT_AVG_PERIOD = 10


def _cdllongline_impl(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Long Line Candle pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdllongline'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    body_long_avg_period = CDLLONGLINE_BODY_LONG_AVG_PERIOD
    shadow_short_avg_period = CDLLONGLINE_SHADOW_SHORT_AVG_PERIOD
    lookback_total = max(body_long_avg_period, shadow_short_avg_period)

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdllongline', values=out))

    body_period_total = CDLLONGLINE_BODY_PERIOD_TOTAL
    shadow_period_total = CDLLONGLINE_SHADOW_PERIOD_TOTAL
    body_trailing_idx = lookback_total - body_long_avg_period
    shadow_trailing_idx = lookback_total - shadow_short_avg_period

    i = body_trailing_idx
    while i < lookback_total:
        body_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = shadow_trailing_idx
    while i < lookback_total:
        upper_shadow = high_values[i] - max(open_values[i], close_values[i])
        lower_shadow = min(open_values[i], close_values[i]) - low_values[i]
        shadow_period_total += upper_shadow + lower_shadow
        i += 1

    i = lookback_total
    while i < n:
        real_body = abs(close_values[i] - open_values[i])
        upper_shadow = high_values[i] - max(open_values[i], close_values[i])
        lower_shadow = min(open_values[i], close_values[i]) - low_values[i]
        color = 1 if close_values[i] >= open_values[i] else -1

        body_long_avg = body_period_total / body_long_avg_period

        shadow_short_avg = (shadow_period_total / shadow_short_avg_period) / 2.0

        if (
            real_body > body_long_avg
            and upper_shadow < shadow_short_avg
            and lower_shadow < shadow_short_avg
        ):
            out[i] = color * 100

        body_period_total += (
            abs(close_values[i] - open_values[i])
            - abs(close_values[body_trailing_idx] - open_values[body_trailing_idx])
        )
        shadow_period_total += (
            (high_values[i] - max(open_values[i], close_values[i]))
            + (min(open_values[i], close_values[i]) - low_values[i])
            - (high_values[shadow_trailing_idx] - max(open_values[shadow_trailing_idx], close_values[shadow_trailing_idx]))
            - (min(open_values[shadow_trailing_idx], close_values[shadow_trailing_idx]) - low_values[shadow_trailing_idx])
        )
        i += 1
        body_trailing_idx += 1
        shadow_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdllongline', values=out))


def cdllongline(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    out_col = 'cdllongline'
    input_cols = [open_col, high_col, low_col, close_col]
    return data.with_columns(
        pl.struct(input_cols).map_batches(
            lambda s: _cdllongline_impl(
                pl.DataFrame({col: s.struct.field(col) for col in input_cols}),
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            ).get_column(out_col),
            return_dtype=pl.Int32,
        ).alias(out_col)
    )
