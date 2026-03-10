import numpy as np
import polars as pl

CDLMATHOLD_BODY_LONG_AVG_PERIOD = 10
CDLMATHOLD_BODY_SHORT_AVG_PERIOD = 10


def _cdlmathold_impl(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    penetration: float = 0.5,
) -> pl.DataFrame:

    '''
    Compute Mat Hold candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices
        penetration (float): Maximum percentage penetration of reaction days into the first white body

    Returns:
        pl.DataFrame: The input data with a new column 'cdlmathold'
    '''

    if penetration < 0.0 or penetration > 3e37:
        raise ValueError('penetration must be between 0 and 3e37')

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    body_short_avg_period = CDLMATHOLD_BODY_SHORT_AVG_PERIOD
    body_long_avg_period = CDLMATHOLD_BODY_LONG_AVG_PERIOD
    lookback_total = max(body_short_avg_period, body_long_avg_period) + 4

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlmathold', values=out))

    body_period_total = np.zeros(5, dtype=float)
    body_short_trailing_idx = lookback_total - body_short_avg_period
    body_long_trailing_idx = lookback_total - body_long_avg_period

    i = body_short_trailing_idx
    while i < lookback_total:
        body_period_total[3] += abs(close_values[i - 3] - open_values[i - 3])
        body_period_total[2] += abs(close_values[i - 2] - open_values[i - 2])
        body_period_total[1] += abs(close_values[i - 1] - open_values[i - 1])
        i += 1

    i = body_long_trailing_idx
    while i < lookback_total:
        body_period_total[4] += abs(close_values[i - 4] - open_values[i - 4])
        i += 1

    i = lookback_total
    while i < n:
        real_body_i4 = abs(close_values[i - 4] - open_values[i - 4])
        real_body_i3 = abs(close_values[i - 3] - open_values[i - 3])
        real_body_i2 = abs(close_values[i - 2] - open_values[i - 2])
        real_body_i1 = abs(close_values[i - 1] - open_values[i - 1])

        body_long_avg_i4 = body_period_total[4] / body_long_avg_period
        body_short_avg_i3 = body_period_total[3] / body_short_avg_period
        body_short_avg_i2 = body_period_total[2] / body_short_avg_period
        body_short_avg_i1 = body_period_total[1] / body_short_avg_period

        color_i4 = 1 if close_values[i - 4] >= open_values[i - 4] else -1
        color_i3 = 1 if close_values[i - 3] >= open_values[i - 3] else -1
        color_i0 = 1 if close_values[i] >= open_values[i] else -1

        second_rb_low = min(open_values[i - 3], close_values[i - 3])
        first_rb_high = max(open_values[i - 4], close_values[i - 4])

        third_rb_low = min(open_values[i - 2], close_values[i - 2])
        fourth_rb_low = min(open_values[i - 1], close_values[i - 1])

        third_rb_high = max(open_values[i - 2], close_values[i - 2])
        fourth_rb_high = max(open_values[i - 1], close_values[i - 1])

        if (
            real_body_i4 > body_long_avg_i4
            and real_body_i3 < body_short_avg_i3
            and real_body_i2 < body_short_avg_i2
            and real_body_i1 < body_short_avg_i1
            and color_i4 == 1
            and color_i3 == -1
            and color_i0 == 1
            and second_rb_low > first_rb_high
            and third_rb_low < close_values[i - 4]
            and fourth_rb_low < close_values[i - 4]
            and third_rb_low > close_values[i - 4] - (real_body_i4 * penetration)
            and fourth_rb_low > close_values[i - 4] - (real_body_i4 * penetration)
            and third_rb_high < open_values[i - 3]
            and fourth_rb_high < third_rb_high
            and open_values[i] > close_values[i - 1]
            and close_values[i] > max(max(high_values[i - 3], high_values[i - 2]), high_values[i - 1])
        ):
            out[i] = 100

        body_period_total[4] += (
            real_body_i4
            - abs(close_values[body_long_trailing_idx - 4] - open_values[body_long_trailing_idx - 4])
        )
        for tot_idx in range(3, 0, -1):
            body_period_total[tot_idx] += (
                abs(close_values[i - tot_idx] - open_values[i - tot_idx])
                - abs(close_values[body_short_trailing_idx - tot_idx] - open_values[body_short_trailing_idx - tot_idx])
            )

        i += 1
        body_short_trailing_idx += 1
        body_long_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdlmathold', values=out))


def cdlmathold(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    penetration: float = 0.5,
) -> pl.DataFrame:

    out_col = 'cdlmathold'
    input_cols = [open_col, high_col, low_col, close_col]
    return data.with_columns(
        pl.struct(input_cols).map_batches(
            lambda s: _cdlmathold_impl(
                pl.DataFrame({col: s.struct.field(col) for col in input_cols}),
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
                penetration=penetration,
            ).get_column(out_col),
            return_dtype=pl.Int32,
        ).alias(out_col)
    )
