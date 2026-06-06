import numpy as np
import polars as pl

CMP_N_3E37 = 3e37

CDLABANDONEDBABY_BODY_DOJI_AVG_PERIOD = 10
CDLABANDONEDBABY_BODY_DOJI_FACTOR = 0.1
CDLABANDONEDBABY_BODY_DOJI_PERIOD_TOTAL = 0.0
CDLABANDONEDBABY_BODY_LONG_AVG_PERIOD = 10
CDLABANDONEDBABY_BODY_LONG_PERIOD_TOTAL = 0.0
CDLABANDONEDBABY_BODY_SHORT_AVG_PERIOD = 10
CDLABANDONEDBABY_BODY_SHORT_PERIOD_TOTAL = 0.0


def _cdlabandonedbaby_impl(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    penetration: float = 0.3,
) -> pl.DataFrame:

    '''
    Compute Abandoned Baby candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices
        penetration (float): Percentage of penetration of the 3rd candle into the 1st real body

    Returns:
        pl.DataFrame: The input data with a new column 'cdlabandonedbaby'
    '''

    if penetration < 0.0 or penetration > CMP_N_3E37:
        raise ValueError('cdlabandonedbaby penetration must be between 0 and 3e37')

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    body_long_avg_period = CDLABANDONEDBABY_BODY_LONG_AVG_PERIOD
    body_short_avg_period = CDLABANDONEDBABY_BODY_SHORT_AVG_PERIOD
    body_doji_avg_period = CDLABANDONEDBABY_BODY_DOJI_AVG_PERIOD
    body_doji_factor = CDLABANDONEDBABY_BODY_DOJI_FACTOR

    lookback_total = max(body_doji_avg_period, body_long_avg_period, body_short_avg_period) + 2

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlabandonedbaby', values=out))

    body_doji_period_total = CDLABANDONEDBABY_BODY_DOJI_PERIOD_TOTAL
    body_long_period_total = CDLABANDONEDBABY_BODY_LONG_PERIOD_TOTAL
    body_short_period_total = CDLABANDONEDBABY_BODY_SHORT_PERIOD_TOTAL

    body_long_trailing_idx = lookback_total - 2 - body_long_avg_period
    body_doji_trailing_idx = lookback_total - 1 - body_doji_avg_period
    body_short_trailing_idx = lookback_total - body_short_avg_period

    i = body_long_trailing_idx
    while i < lookback_total - 2:
        body_long_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = body_doji_trailing_idx
    while i < lookback_total - 1:
        body_doji_period_total += high_values[i] - low_values[i]
        i += 1

    i = body_short_trailing_idx
    while i < lookback_total:
        body_short_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = lookback_total
    while i < n:
        real_body_i2 = abs(close_values[i - 2] - open_values[i - 2])
        real_body_i1 = abs(close_values[i - 1] - open_values[i - 1])
        real_body_i0 = abs(close_values[i] - open_values[i])

        body_long_avg_i2 = body_long_period_total / body_long_avg_period
        body_doji_avg_i1 = body_doji_factor * (body_doji_period_total / body_doji_avg_period)
        body_short_avg_i0 = body_short_period_total / body_short_avg_period

        color_i2 = 1 if close_values[i - 2] >= open_values[i - 2] else -1
        color_i0 = 1 if close_values[i] >= open_values[i] else -1

        candle_gap_up_1_2 = low_values[i - 1] > high_values[i - 2]
        candle_gap_down_1_2 = high_values[i - 1] < low_values[i - 2]
        candle_gap_up_0_1 = low_values[i] > high_values[i - 1]
        candle_gap_down_0_1 = high_values[i] < low_values[i - 1]

        if (
            real_body_i2 > body_long_avg_i2
            and real_body_i1 <= body_doji_avg_i1
            and real_body_i0 > body_short_avg_i0
            and (
                (
                    color_i2 == 1
                    and color_i0 == -1
                    and close_values[i] < close_values[i - 2] - (real_body_i2 * penetration)
                    and candle_gap_up_1_2
                    and candle_gap_down_0_1
                )
                or (
                    color_i2 == -1
                    and color_i0 == 1
                    and close_values[i] > close_values[i - 2] + (real_body_i2 * penetration)
                    and candle_gap_down_1_2
                    and candle_gap_up_0_1
                )
            )
        ):
            out[i] = color_i0 * 100

        body_long_period_total += (
            abs(close_values[i - 2] - open_values[i - 2])
            - abs(close_values[body_long_trailing_idx] - open_values[body_long_trailing_idx])
        )
        body_doji_period_total += (
            (high_values[i - 1] - low_values[i - 1])
            - (high_values[body_doji_trailing_idx] - low_values[body_doji_trailing_idx])
        )
        body_short_period_total += (
            abs(close_values[i] - open_values[i])
            - abs(close_values[body_short_trailing_idx] - open_values[body_short_trailing_idx])
        )

        i += 1
        body_long_trailing_idx += 1
        body_doji_trailing_idx += 1
        body_short_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdlabandonedbaby', values=out))


def cdlabandonedbaby(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    penetration: float = 0.3,
) -> pl.DataFrame:

    out_col = 'cdlabandonedbaby'
    input_cols = [open_col, high_col, low_col, close_col]
    return data.with_columns(
        pl.struct(input_cols).map_batches(
            lambda s: _cdlabandonedbaby_impl(
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
