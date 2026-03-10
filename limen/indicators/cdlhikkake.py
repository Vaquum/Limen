import numpy as np
import polars as pl


def _cdlhikkake_impl(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Hikkake candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdlhikkake'
    '''

    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    lookback_total = 5

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlhikkake', values=out))

    start_idx = lookback_total
    pattern_idx = 0
    pattern_result = 0

    i = start_idx - 3
    while i < start_idx:
        if (
            high_values[i - 1] < high_values[i - 2]
            and low_values[i - 1] > low_values[i - 2]
            and (
                (high_values[i] < high_values[i - 1] and low_values[i] < low_values[i - 1])
                or (high_values[i] > high_values[i - 1] and low_values[i] > low_values[i - 1])
            )
        ):
            pattern_result = 100 * (1 if high_values[i] < high_values[i - 1] else -1)
            pattern_idx = i
        elif (
            i <= pattern_idx + 3
            and (
                (pattern_result > 0 and close_values[i] > high_values[pattern_idx - 1])
                or (pattern_result < 0 and close_values[i] < low_values[pattern_idx - 1])
            )
        ):
            pattern_idx = 0
        i += 1

    i = start_idx
    while i < n:
        if (
            high_values[i - 1] < high_values[i - 2]
            and low_values[i - 1] > low_values[i - 2]
            and (
                (high_values[i] < high_values[i - 1] and low_values[i] < low_values[i - 1])
                or (high_values[i] > high_values[i - 1] and low_values[i] > low_values[i - 1])
            )
        ):
            pattern_result = 100 * (1 if high_values[i] < high_values[i - 1] else -1)
            pattern_idx = i
            out[i] = pattern_result
        elif (
            i <= pattern_idx + 3
            and (
                (pattern_result > 0 and close_values[i] > high_values[pattern_idx - 1])
                or (pattern_result < 0 and close_values[i] < low_values[pattern_idx - 1])
            )
        ):
            out[i] = pattern_result + (100 if pattern_result > 0 else -100)
            pattern_idx = 0

        i += 1

    return data.with_columns(pl.Series(name='cdlhikkake', values=out))


def cdlhikkake(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    out_col = 'cdlhikkake'
    input_cols = [open_col, high_col, low_col, close_col]
    return data.with_columns(
        pl.struct(input_cols).map_batches(
            lambda s: _cdlhikkake_impl(
                pl.DataFrame({col: s.struct.field(col) for col in input_cols}),
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            ).get_column(out_col),
            return_dtype=pl.Int32,
        ).alias(out_col)
    )
