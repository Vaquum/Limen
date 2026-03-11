import numpy as np
import polars as pl

CDLSEPARATINGLINES_BODY_LONG_AVG_PERIOD = 10
CDLSEPARATINGLINES_BODY_LONG_PERIOD_TOTAL = 0.0
CDLSEPARATINGLINES_EQUAL_AVG_PERIOD = 5
CDLSEPARATINGLINES_EQUAL_FACTOR = 0.05
CDLSEPARATINGLINES_EQUAL_PERIOD_TOTAL = 0.0
CDLSEPARATINGLINES_SHADOW_VS_AVG_PERIOD = 10
CDLSEPARATINGLINES_SHADOW_VS_FACTOR = 0.1
CDLSEPARATINGLINES_SHADOW_VS_PERIOD_TOTAL = 0.0


def _cdlseparatinglines_impl(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Separating Lines candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdlseparatinglines'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    shadow_vs_avg_period = CDLSEPARATINGLINES_SHADOW_VS_AVG_PERIOD
    shadow_vs_factor = CDLSEPARATINGLINES_SHADOW_VS_FACTOR
    body_long_avg_period = CDLSEPARATINGLINES_BODY_LONG_AVG_PERIOD
    equal_avg_period = CDLSEPARATINGLINES_EQUAL_AVG_PERIOD
    equal_factor = CDLSEPARATINGLINES_EQUAL_FACTOR
    lookback_total = max(shadow_vs_avg_period, body_long_avg_period, equal_avg_period) + 1

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlseparatinglines', values=out))

    shadow_vs_period_total = CDLSEPARATINGLINES_SHADOW_VS_PERIOD_TOTAL
    body_long_period_total = CDLSEPARATINGLINES_BODY_LONG_PERIOD_TOTAL
    equal_period_total = CDLSEPARATINGLINES_EQUAL_PERIOD_TOTAL
    shadow_vs_trailing_idx = lookback_total - shadow_vs_avg_period
    body_long_trailing_idx = lookback_total - body_long_avg_period
    equal_trailing_idx = lookback_total - equal_avg_period

    i = shadow_vs_trailing_idx
    while i < lookback_total:
        shadow_vs_period_total += high_values[i] - low_values[i]
        i += 1

    i = body_long_trailing_idx
    while i < lookback_total:
        body_long_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = equal_trailing_idx
    while i < lookback_total:
        equal_period_total += high_values[i - 1] - low_values[i - 1]
        i += 1

    i = lookback_total
    while i < n:
        color_i1 = 1 if close_values[i - 1] >= open_values[i - 1] else -1
        color_i0 = 1 if close_values[i] >= open_values[i] else -1
        real_body_i0 = abs(close_values[i] - open_values[i])
        upper_shadow_i0 = high_values[i] - max(open_values[i], close_values[i])
        lower_shadow_i0 = min(open_values[i], close_values[i]) - low_values[i]

        equal_avg_i1 = equal_factor * (equal_period_total / equal_avg_period)
        body_long_avg_i0 = body_long_period_total / body_long_avg_period
        shadow_vs_avg_i0 = shadow_vs_factor * (shadow_vs_period_total / shadow_vs_avg_period)

        if (
            color_i1 == -color_i0
            and open_values[i] <= open_values[i - 1] + equal_avg_i1
            and open_values[i] >= open_values[i - 1] - equal_avg_i1
            and real_body_i0 > body_long_avg_i0
            and (
                (color_i0 == 1 and lower_shadow_i0 < shadow_vs_avg_i0)
                or (color_i0 == -1 and upper_shadow_i0 < shadow_vs_avg_i0)
            )
        ):
            out[i] = color_i0 * 100

        shadow_vs_period_total += (
            (high_values[i] - low_values[i])
            - (high_values[shadow_vs_trailing_idx] - low_values[shadow_vs_trailing_idx])
        )
        body_long_period_total += (
            abs(close_values[i] - open_values[i])
            - abs(close_values[body_long_trailing_idx] - open_values[body_long_trailing_idx])
        )
        equal_period_total += (
            (high_values[i - 1] - low_values[i - 1])
            - (high_values[equal_trailing_idx - 1] - low_values[equal_trailing_idx - 1])
        )

        i += 1
        shadow_vs_trailing_idx += 1
        body_long_trailing_idx += 1
        equal_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdlseparatinglines', values=out))


def cdlseparatinglines(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    out_col = 'cdlseparatinglines'
    input_cols = [open_col, high_col, low_col, close_col]
    return data.with_columns(
        pl.struct(input_cols).map_batches(
            lambda s: _cdlseparatinglines_impl(
                pl.DataFrame({col: s.struct.field(col) for col in input_cols}),
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            ).get_column(out_col),
            return_dtype=pl.Int32,
        ).alias(out_col)
    )
