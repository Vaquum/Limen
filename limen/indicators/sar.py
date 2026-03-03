import numpy as np
import polars as pl


def sar(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    acceleration: float = 0.02,
    maximum: float = 0.2,
) -> pl.DataFrame:

    '''
    Compute Parabolic SAR.

    Args:
        data (pl.DataFrame): Dataset with high/low columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        acceleration (float): Acceleration factor
        maximum (float): Maximum acceleration factor

    Returns:
        pl.DataFrame: The input data with a new column 'sar'
    '''

    if acceleration < 0.0 or acceleration > 3e37:
        raise ValueError('acceleration must be between 0 and 3e37')
    if maximum < 0.0 or maximum > 3e37:
        raise ValueError('maximum must be between 0 and 3e37')

    high = data[high_col].to_numpy().astype(float, copy=False)
    low = data[low_col].to_numpy().astype(float, copy=False)
    n = len(data)

    out = np.full(n, np.nan, dtype=float)
    if n <= 1:
        return data.with_columns(pl.Series(name='sar', values=out))

    start_idx = 1
    end_idx = n - 1

    af = acceleration
    if af > maximum:
        af = acceleration = maximum


    diff_p = high[start_idx] - high[start_idx - 1]
    diff_m = low[start_idx - 1] - low[start_idx]
    is_long = 0 if ((diff_m > 0.0) and (diff_p < diff_m)) else 1

    today_idx = start_idx

    new_high = high[today_idx - 1]
    new_low = low[today_idx - 1]

    if is_long == 1:
        ep = high[today_idx]
        sar_value = new_low
    else:
        ep = low[today_idx]
        sar_value = new_high

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

                out[out_series_idx] = sar_value
                out_series_idx += 1

                af = acceleration
                ep = new_low

                sar_value = sar_value + (af * (ep - sar_value))

                if sar_value < prev_high:
                    sar_value = prev_high
                if sar_value < new_high:
                    sar_value = new_high
            else:
                out[out_series_idx] = sar_value
                out_series_idx += 1

                if new_high > ep:
                    ep = new_high
                    af += acceleration
                    if af > maximum:
                        af = maximum

                sar_value = sar_value + (af * (ep - sar_value))

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

                out[out_series_idx] = sar_value
                out_series_idx += 1

                af = acceleration
                ep = new_high

                sar_value = sar_value + (af * (ep - sar_value))

                if sar_value > prev_low:
                    sar_value = prev_low
                if sar_value > new_low:
                    sar_value = new_low
            else:
                out[out_series_idx] = sar_value
                out_series_idx += 1

                if new_low < ep:
                    ep = new_low
                    af += acceleration
                    if af > maximum:
                        af = maximum

                sar_value = sar_value + (af * (ep - sar_value))

                if sar_value < prev_high:
                    sar_value = prev_high
                if sar_value < new_high:
                    sar_value = new_high

    return data.with_columns(pl.Series(name='sar', values=out))
