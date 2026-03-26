import numpy as np
import polars as pl

CDLHARAMI_BODY_LONG_AVG_PERIOD = 10
CDLHARAMI_BODY_LONG_PERIOD_TOTAL = 0.0
CDLHARAMI_BODY_SHORT_AVG_PERIOD = 10
CDLHARAMI_BODY_SHORT_PERIOD_TOTAL = 0.0


def _cdlharami_impl(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Harami candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdlharami'
    '''
    _ = high_col, low_col

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    body_long_avg_period = CDLHARAMI_BODY_LONG_AVG_PERIOD
    body_short_avg_period = CDLHARAMI_BODY_SHORT_AVG_PERIOD
    lookback_total = max(body_short_avg_period, body_long_avg_period) + 1

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlharami', values=out))

    body_short_period_total = CDLHARAMI_BODY_SHORT_PERIOD_TOTAL
    body_long_period_total = CDLHARAMI_BODY_LONG_PERIOD_TOTAL
    body_long_trailing_idx = lookback_total - 1 - body_long_avg_period
    body_short_trailing_idx = lookback_total - body_short_avg_period

    i = body_long_trailing_idx
    while i < lookback_total - 1:
        body_long_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = body_short_trailing_idx
    while i < lookback_total:
        body_short_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = lookback_total
    while i < n:
        real_body_i1 = abs(close_values[i - 1] - open_values[i - 1])
        real_body_i0 = abs(close_values[i] - open_values[i])
        body_long_avg_i1 = body_long_period_total / body_long_avg_period
        body_short_avg_i0 = body_short_period_total / body_short_avg_period

        if real_body_i1 > body_long_avg_i1 and real_body_i0 <= body_short_avg_i0:
            high_i0 = max(close_values[i], open_values[i])
            low_i0 = min(close_values[i], open_values[i])
            high_i1 = max(close_values[i - 1], open_values[i - 1])
            low_i1 = min(close_values[i - 1], open_values[i - 1])
            color_i1 = 1 if close_values[i - 1] >= open_values[i - 1] else -1

            if high_i0 < high_i1 and low_i0 > low_i1:
                out[i] = -color_i1 * 100
            elif high_i0 <= high_i1 and low_i0 >= low_i1:
                out[i] = -color_i1 * 80

        body_long_period_total += (
            abs(close_values[i - 1] - open_values[i - 1])
            - abs(close_values[body_long_trailing_idx] - open_values[body_long_trailing_idx])
        )
        body_short_period_total += (
            abs(close_values[i] - open_values[i])
            - abs(close_values[body_short_trailing_idx] - open_values[body_short_trailing_idx])
        )
        i += 1
        body_long_trailing_idx += 1
        body_short_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdlharami', values=out))


def cdlharami(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    out_col = 'cdlharami'
    input_cols = [open_col, high_col, low_col, close_col]
    return data.with_columns(
        pl.struct(input_cols).map_batches(
            lambda s: _cdlharami_impl(
                pl.DataFrame({col: s.struct.field(col) for col in input_cols}),
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            ).get_column(out_col),
            return_dtype=pl.Int32,
        ).alias(out_col)
    )
