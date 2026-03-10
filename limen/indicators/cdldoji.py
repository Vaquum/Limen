import numpy as np
import polars as pl

COLDOJI_BODY_DOJI_AVG_PERIOD = 10
COLDOJI_BODY_DOJI_FACTOR = 0.1
COLDOJI_BODY_DOJI_PERIOD_TOTAL = 0.0


def _coldoji_impl(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Doji candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'coldoji'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    body_doji_avg_period = COLDOJI_BODY_DOJI_AVG_PERIOD
    body_doji_factor = COLDOJI_BODY_DOJI_FACTOR
    lookback_total = body_doji_avg_period

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='coldoji', values=out))

    body_doji_period_total = COLDOJI_BODY_DOJI_PERIOD_TOTAL
    body_doji_trailing_idx = lookback_total - body_doji_avg_period

    i = body_doji_trailing_idx
    while i < lookback_total:
        body_doji_period_total += high_values[i] - low_values[i]
        i += 1

    i = lookback_total
    while i < n:
        real_body = abs(close_values[i] - open_values[i])
        body_doji_avg = body_doji_factor * (body_doji_period_total / body_doji_avg_period)

        if real_body <= body_doji_avg:
            out[i] = 100

        body_doji_period_total += (
            (high_values[i] - low_values[i])
            - (high_values[body_doji_trailing_idx] - low_values[body_doji_trailing_idx])
        )
        i += 1
        body_doji_trailing_idx += 1

    return data.with_columns(pl.Series(name='coldoji', values=out))


def coldoji(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    out_col = 'coldoji'
    input_cols = [open_col, high_col, low_col, close_col]
    return data.with_columns(
        pl.struct(input_cols).map_batches(
            lambda s: _coldoji_impl(
                pl.DataFrame({col: s.struct.field(col) for col in input_cols}),
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            ).get_column(out_col),
            return_dtype=pl.Int32,
        ).alias(out_col)
    )
