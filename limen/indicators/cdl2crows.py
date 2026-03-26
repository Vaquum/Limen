import numpy as np
import polars as pl

CDL2CROWS_BODY_LONG_AVG_PERIOD = 10
CDL2CROWS_BODY_LONG_PERIOD_TOTAL = 0.0


def _cdl2crows_impl(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Two Crows candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdl2crows'
    '''
    _ = high_col, low_col

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    body_long_avg_period = CDL2CROWS_BODY_LONG_AVG_PERIOD
    lookback_total = body_long_avg_period + 2

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdl2crows', values=out))

    start_idx = lookback_total
    body_long_period_total = CDL2CROWS_BODY_LONG_PERIOD_TOTAL
    body_long_trailing_idx = start_idx - 2 - body_long_avg_period

    i = body_long_trailing_idx
    while i < start_idx - 2:
        body_long_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = start_idx
    while i < n:
        real_body_i2 = abs(close_values[i - 2] - open_values[i - 2])
        body_long_average_i2 = body_long_period_total / body_long_avg_period

        first_white = close_values[i - 2] >= open_values[i - 2]
        second_black = close_values[i - 1] < open_values[i - 1]
        third_black = close_values[i] < open_values[i]

        second_rb_low = min(open_values[i - 1], close_values[i - 1])
        max(open_values[i - 1], close_values[i - 1])
        first_rb_high = max(open_values[i - 2], close_values[i - 2])
        gap_up = second_rb_low > first_rb_high

        third_opens_within_second_rb = (
            open_values[i] < open_values[i - 1]
            and open_values[i] > close_values[i - 1]
        )
        third_closes_within_first_rb = (
            close_values[i] > open_values[i - 2]
            and close_values[i] < close_values[i - 2]
        )

        if (
            first_white
            and real_body_i2 > body_long_average_i2
            and second_black
            and gap_up
            and third_black
            and third_opens_within_second_rb
            and third_closes_within_first_rb
        ):
            out[i] = -100

        body_long_period_total += (
            abs(close_values[i - 2] - open_values[i - 2])
            - abs(close_values[body_long_trailing_idx] - open_values[body_long_trailing_idx])
        )
        body_long_trailing_idx += 1
        i += 1

    return data.with_columns(pl.Series(name='cdl2crows', values=out))


def cdl2crows(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    out_col = 'cdl2crows'
    input_cols = [open_col, high_col, low_col, close_col]
    return data.with_columns(
        pl.struct(input_cols).map_batches(
            lambda s: _cdl2crows_impl(
                pl.DataFrame({col: s.struct.field(col) for col in input_cols}),
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            ).get_column(out_col),
            return_dtype=pl.Int32,
        ).alias(out_col)
    )
