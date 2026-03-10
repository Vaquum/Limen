import numpy as np
import polars as pl

CDL3LINESTRIKE_NEAR_AVG_PERIOD = 5
CDL3LINESTRIKE_NEAR_FACTOR = 0.2


def _cdl3linestrike_impl(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Three-Line Strike candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdl3linestrike'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    near_avg_period = CDL3LINESTRIKE_NEAR_AVG_PERIOD
    near_factor = CDL3LINESTRIKE_NEAR_FACTOR
    lookback_total = near_avg_period + 3

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdl3linestrike', values=out))

    near_period_total = np.zeros(4, dtype=float)
    near_trailing_idx = lookback_total - near_avg_period

    i = near_trailing_idx
    while i < lookback_total:
        near_period_total[3] += high_values[i - 3] - low_values[i - 3]
        near_period_total[2] += high_values[i - 2] - low_values[i - 2]
        i += 1

    i = lookback_total
    while i < n:
        color_i3 = 1 if close_values[i - 3] >= open_values[i - 3] else -1
        color_i2 = 1 if close_values[i - 2] >= open_values[i - 2] else -1
        color_i1 = 1 if close_values[i - 1] >= open_values[i - 1] else -1
        color_i0 = 1 if close_values[i] >= open_values[i] else -1

        near_avg_i3 = near_factor * (near_period_total[3] / near_avg_period)
        near_avg_i2 = near_factor * (near_period_total[2] / near_avg_period)

        if (
            color_i3 == color_i2
            and color_i2 == color_i1
            and color_i0 == -color_i1
            and open_values[i - 2] >= min(open_values[i - 3], close_values[i - 3]) - near_avg_i3
            and open_values[i - 2] <= max(open_values[i - 3], close_values[i - 3]) + near_avg_i3
            and open_values[i - 1] >= min(open_values[i - 2], close_values[i - 2]) - near_avg_i2
            and open_values[i - 1] <= max(open_values[i - 2], close_values[i - 2]) + near_avg_i2
            and (
                (
                    color_i1 == 1
                    and close_values[i - 1] > close_values[i - 2]
                    and close_values[i - 2] > close_values[i - 3]
                    and open_values[i] > close_values[i - 1]
                    and close_values[i] < open_values[i - 3]
                )
                or (
                    color_i1 == -1
                    and close_values[i - 1] < close_values[i - 2]
                    and close_values[i - 2] < close_values[i - 3]
                    and open_values[i] < close_values[i - 1]
                    and close_values[i] > open_values[i - 3]
                )
            )
        ):
            out[i] = color_i1 * 100

        for tot_idx in range(3, 1, -1):
            near_period_total[tot_idx] += (
                (high_values[i - tot_idx] - low_values[i - tot_idx])
                - (high_values[near_trailing_idx - tot_idx] - low_values[near_trailing_idx - tot_idx])
            )

        i += 1
        near_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdl3linestrike', values=out))

def cdl3linestrike(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    out_col = 'cdl3linestrike'
    input_cols = [open_col, high_col, low_col, close_col]
    return data.with_columns(
        pl.struct(input_cols).map_batches(
            lambda s: _cdl3linestrike_impl(
                pl.DataFrame({col: s.struct.field(col) for col in input_cols}),
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            ).get_column(out_col),
            return_dtype=pl.Int32,
        ).alias(out_col)
    )
