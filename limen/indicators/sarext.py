import numpy as np
import polars as pl


def sarext(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    start_value: float = 0.0,
    offset_on_reverse: float = 0.0,
    acceleration_init_long: float = 0.02,
    acceleration_long: float = 0.02,
    acceleration_max_long: float = 0.2,
    acceleration_init_short: float = 0.02,
    acceleration_short: float = 0.02,
    acceleration_max_short: float = 0.2,
) -> pl.DataFrame:

    '''
    Compute Parabolic SAR - Extended (SAREXT).

    Args:
        data (pl.DataFrame): Dataset with high/low columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        start_value (float): Start value and direction
        offset_on_reverse (float): Percent offset on reversal
        acceleration_init_long (float): Initial AF for long
        acceleration_long (float): AF increment for long
        acceleration_max_long (float): AF max for long
        acceleration_init_short (float): Initial AF for short
        acceleration_short (float): AF increment for short
        acceleration_max_short (float): AF max for short

    Returns:
        pl.DataFrame: The input data with a new column 'sarext'
    '''

    if start_value < -3e37 or start_value > 3e37:
        raise ValueError('start_value must be between -3e37 and 3e37')
    if offset_on_reverse < 0.0 or offset_on_reverse > 3e37:
        raise ValueError('offset_on_reverse must be between 0 and 3e37')
    if acceleration_init_long < 0.0 or acceleration_init_long > 3e37:
        raise ValueError('acceleration_init_long must be between 0 and 3e37')
    if acceleration_long < 0.0 or acceleration_long > 3e37:
        raise ValueError('acceleration_long must be between 0 and 3e37')
    if acceleration_max_long < 0.0 or acceleration_max_long > 3e37:
        raise ValueError('acceleration_max_long must be between 0 and 3e37')
    if acceleration_init_short < 0.0 or acceleration_init_short > 3e37:
        raise ValueError('acceleration_init_short must be between 0 and 3e37')
    if acceleration_short < 0.0 or acceleration_short > 3e37:
        raise ValueError('acceleration_short must be between 0 and 3e37')
    if acceleration_max_short < 0.0 or acceleration_max_short > 3e37:
        raise ValueError('acceleration_max_short must be between 0 and 3e37')

    high = data[high_col].to_numpy().astype(float, copy=False)
    low = data[low_col].to_numpy().astype(float, copy=False)
    n = len(data)

    out = np.full(n, np.nan, dtype=float)
    if n <= 1:
        return data.with_columns(pl.Series(name='sarext', values=out))

    start_idx = 1
    end_idx = n - 1

    af_long = acceleration_init_long
    af_short = acceleration_init_short

    if af_long > acceleration_max_long:
        af_long = acceleration_max_long
        acceleration_init_long = acceleration_max_long
    if acceleration_long > acceleration_max_long:
        acceleration_long = acceleration_max_long

    if af_short > acceleration_max_short:
        af_short = acceleration_max_short
        acceleration_init_short = acceleration_max_short
    if acceleration_short > acceleration_max_short:
        acceleration_short = acceleration_max_short

    if start_value == 0.0:

        diff_p = high[start_idx] - high[start_idx - 1]
        diff_m = low[start_idx - 1] - low[start_idx]
        is_long = 0 if ((diff_m > 0.0) and (diff_p < diff_m)) else 1
    elif start_value > 0.0:
        is_long = 1
    else:
        is_long = 0

    today_idx = start_idx

    new_high = high[today_idx - 1]
    new_low = low[today_idx - 1]

    if start_value == 0.0:
        if is_long == 1:
            ep = high[today_idx]
            sar_value = new_low
        else:
            ep = low[today_idx]
            sar_value = new_high
    elif start_value > 0.0:
        ep = high[today_idx]
        sar_value = start_value
    else:
        ep = low[today_idx]
        sar_value = abs(start_value)

    new_low = low[today_idx]
    new_high = high[today_idx]

    out_series_idx = start_idx
    while today_idx <= end_idx:
        prev_low = new_low
        prev_high = new_high
        new_low = low[today_idx]
        new_high = high[today_idx]
        today_idx += 1

        if is_long == 1:
            if new_low <= sar_value:
                is_long = 0
                sar_value = ep

                if sar_value < prev_high:
                    sar_value = prev_high
                if sar_value < new_high:
                    sar_value = new_high

                if offset_on_reverse != 0.0:
                    sar_value += sar_value * offset_on_reverse
                out[out_series_idx] = -sar_value
                out_series_idx += 1

                af_short = acceleration_init_short
                ep = new_low

                sar_value = sar_value + (af_short * (ep - sar_value))

                if sar_value < prev_high:
                    sar_value = prev_high
                if sar_value < new_high:
                    sar_value = new_high
            else:
                out[out_series_idx] = sar_value
                out_series_idx += 1

                if new_high > ep:
                    ep = new_high
                    af_long += acceleration_long
                    if af_long > acceleration_max_long:
                        af_long = acceleration_max_long

                sar_value = sar_value + (af_long * (ep - sar_value))

                if sar_value > prev_low:
                    sar_value = prev_low
                if sar_value > new_low:
                    sar_value = new_low
        else:
            if new_high >= sar_value:
                is_long = 1
                sar_value = ep

                if sar_value > prev_low:
                    sar_value = prev_low
                if sar_value > new_low:
                    sar_value = new_low

                if offset_on_reverse != 0.0:
                    sar_value -= sar_value * offset_on_reverse
                out[out_series_idx] = sar_value
                out_series_idx += 1

                af_long = acceleration_init_long
                ep = new_high

                sar_value = sar_value + (af_long * (ep - sar_value))

                if sar_value > prev_low:
                    sar_value = prev_low
                if sar_value > new_low:
                    sar_value = new_low
            else:
                out[out_series_idx] = -sar_value
                out_series_idx += 1

                if new_low < ep:
                    ep = new_low
                    af_short += acceleration_short
                    if af_short > acceleration_max_short:
                        af_short = acceleration_max_short

                sar_value = sar_value + (af_short * (ep - sar_value))

                if sar_value < prev_high:
                    sar_value = prev_high
                if sar_value < new_high:
                    sar_value = new_high

    return data.with_columns(pl.Series(name='sarext', values=out))
