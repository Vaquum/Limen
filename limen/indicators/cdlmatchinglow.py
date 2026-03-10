import numpy as np
import polars as pl

CDLMATCHINGLOW_EQUAL_AVG_PERIOD = 5
CDLMATCHINGLOW_EQUAL_FACTOR = 0.05
CDLMATCHINGLOW_EQUAL_PERIOD_TOTAL = 0.0


def _cdlmatchinglow_impl(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Matching Low candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdlmatchinglow'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    equal_avg_period = CDLMATCHINGLOW_EQUAL_AVG_PERIOD
    equal_factor = CDLMATCHINGLOW_EQUAL_FACTOR
    lookback_total = equal_avg_period + 1

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlmatchinglow', values=out))

    equal_period_total = CDLMATCHINGLOW_EQUAL_PERIOD_TOTAL
    equal_trailing_idx = lookback_total - equal_avg_period

    i = equal_trailing_idx
    while i < lookback_total:
        equal_period_total += high_values[i - 1] - low_values[i - 1]
        i += 1

    i = lookback_total
    while i < n:
        color_i1 = 1 if close_values[i - 1] >= open_values[i - 1] else -1
        color_i0 = 1 if close_values[i] >= open_values[i] else -1
        equal_avg_i1 = equal_factor * (equal_period_total / equal_avg_period)

        if (
            color_i1 == -1
            and color_i0 == -1
            and close_values[i] <= close_values[i - 1] + equal_avg_i1
            and close_values[i] >= close_values[i - 1] - equal_avg_i1
        ):
            out[i] = 100

        equal_period_total += (
            (high_values[i - 1] - low_values[i - 1])
            - (high_values[equal_trailing_idx - 1] - low_values[equal_trailing_idx - 1])
        )
        i += 1
        equal_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdlmatchinglow', values=out))


def cdlmatchinglow(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    out_col = 'cdlmatchinglow'
    input_cols = [open_col, high_col, low_col, close_col]
    return data.with_columns(
        pl.struct(input_cols).map_batches(
            lambda s: _cdlmatchinglow_impl(
                pl.DataFrame({col: s.struct.field(col) for col in input_cols}),
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            ).get_column(out_col),
            return_dtype=pl.Int32,
        ).alias(out_col)
    )
