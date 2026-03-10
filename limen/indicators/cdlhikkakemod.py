import numpy as np
import polars as pl

CDLHIKKAKEMOD_NEAR_AVG_PERIOD = 5
CDLHIKKAKEMOD_NEAR_FACTOR = 0.2
CDLHIKKAKEMOD_NEAR_PERIOD_TOTAL = 0.0


def _cdlhikkakemod_impl(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Modified Hikkake candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdlhikkakemod'
    '''

    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    near_avg_period = CDLHIKKAKEMOD_NEAR_AVG_PERIOD
    near_factor = CDLHIKKAKEMOD_NEAR_FACTOR
    lookback_total = max(1, near_avg_period) + 5

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlhikkakemod', values=out))

    near_period_total = CDLHIKKAKEMOD_NEAR_PERIOD_TOTAL
    near_trailing_idx = lookback_total - 3 - near_avg_period

    i = near_trailing_idx
    while i < lookback_total - 3:
        near_period_total += high_values[i - 2] - low_values[i - 2]
        i += 1

    pattern_idx = 0
    pattern_result = 0

    i = lookback_total - 3
    while i < lookback_total:
        near_avg_i2 = near_factor * (near_period_total / near_avg_period)
        bull_pattern = (
            high_values[i] < high_values[i - 1]
            and low_values[i] < low_values[i - 1]
            and close_values[i - 2] <= low_values[i - 2] + near_avg_i2
        )
        bear_pattern = (
            high_values[i] > high_values[i - 1]
            and low_values[i] > low_values[i - 1]
            and close_values[i - 2] >= high_values[i - 2] - near_avg_i2
        )

        if (
            high_values[i - 2] < high_values[i - 3]
            and low_values[i - 2] > low_values[i - 3]
            and high_values[i - 1] < high_values[i - 2]
            and low_values[i - 1] > low_values[i - 2]
            and (bull_pattern or bear_pattern)
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

        near_period_total += (
            (high_values[i - 2] - low_values[i - 2])
            - (high_values[near_trailing_idx - 2] - low_values[near_trailing_idx - 2])
        )
        near_trailing_idx += 1
        i += 1

    i = lookback_total
    while i < n:
        near_avg_i2 = near_factor * (near_period_total / near_avg_period)
        bull_pattern = (
            high_values[i] < high_values[i - 1]
            and low_values[i] < low_values[i - 1]
            and close_values[i - 2] <= low_values[i - 2] + near_avg_i2
        )
        bear_pattern = (
            high_values[i] > high_values[i - 1]
            and low_values[i] > low_values[i - 1]
            and close_values[i - 2] >= high_values[i - 2] - near_avg_i2
        )

        if (
            high_values[i - 2] < high_values[i - 3]
            and low_values[i - 2] > low_values[i - 3]
            and high_values[i - 1] < high_values[i - 2]
            and low_values[i - 1] > low_values[i - 2]
            and (bull_pattern or bear_pattern)
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

        near_period_total += (
            (high_values[i - 2] - low_values[i - 2])
            - (high_values[near_trailing_idx - 2] - low_values[near_trailing_idx - 2])
        )
        near_trailing_idx += 1
        i += 1

    return data.with_columns(pl.Series(name='cdlhikkakemod', values=out))


def cdlhikkakemod(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    out_col = 'cdlhikkakemod'
    input_cols = [open_col, high_col, low_col, close_col]
    return data.with_columns(
        pl.struct(input_cols).map_batches(
            lambda s: _cdlhikkakemod_impl(
                pl.DataFrame({col: s.struct.field(col) for col in input_cols}),
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            ).get_column(out_col),
            return_dtype=pl.Int32,
        ).alias(out_col)
    )
