import numpy as np
import polars as pl

CDLTRISTAR_BODY_DOJI_AVG_PERIOD = 10
CDLTRISTAR_BODY_DOJI_FACTOR = 0.1
CDLTRISTAR_BODY_PERIOD_TOTAL = 0.0


def _cdltristar_impl(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Tristar candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdltristar'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    body_doji_avg_period = CDLTRISTAR_BODY_DOJI_AVG_PERIOD
    body_doji_factor = CDLTRISTAR_BODY_DOJI_FACTOR
    lookback_total = body_doji_avg_period + 2

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdltristar', values=out))

    body_period_total = CDLTRISTAR_BODY_PERIOD_TOTAL
    body_trailing_idx = lookback_total - 2 - body_doji_avg_period

    i = body_trailing_idx
    while i < lookback_total - 2:
        body_period_total += high_values[i] - low_values[i]
        i += 1

    i = lookback_total
    while i < n:
        body_doji_avg = body_doji_factor * (body_period_total / body_doji_avg_period)
        real_body_i2 = abs(close_values[i - 2] - open_values[i - 2])
        real_body_i1 = abs(close_values[i - 1] - open_values[i - 1])
        real_body_i0 = abs(close_values[i] - open_values[i])

        if (
            real_body_i2 <= body_doji_avg
            and real_body_i1 <= body_doji_avg
            and real_body_i0 <= body_doji_avg
        ):
            real_body_gap_up = min(open_values[i - 1], close_values[i - 1]) > max(open_values[i - 2], close_values[i - 2])
            real_body_gap_down = max(open_values[i - 1], close_values[i - 1]) < min(open_values[i - 2], close_values[i - 2])

            if (
                real_body_gap_up
                and max(open_values[i], close_values[i]) < max(open_values[i - 1], close_values[i - 1])
            ):
                out[i] = -100
            if (
                real_body_gap_down
                and min(open_values[i], close_values[i]) > min(open_values[i - 1], close_values[i - 1])
            ):
                out[i] = 100

        body_period_total += (
            (high_values[i - 2] - low_values[i - 2])
            - (high_values[body_trailing_idx] - low_values[body_trailing_idx])
        )
        i += 1
        body_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdltristar', values=out))


def cdltristar(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    out_col = 'cdltristar'
    input_cols = [open_col, high_col, low_col, close_col]
    return data.with_columns(
        pl.struct(input_cols).map_batches(
            lambda s: _cdltristar_impl(
                pl.DataFrame({col: s.struct.field(col) for col in input_cols}),
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            ).get_column(out_col),
            return_dtype=pl.Int32,
        ).alias(out_col)
    )
