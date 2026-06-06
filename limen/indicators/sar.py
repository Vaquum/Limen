import numpy as np
import polars as pl


CMP_N_3E37 = 3e37

def _sar_from_arrays(
    high: np.ndarray,
    low: np.ndarray,
    acceleration: float,
    maximum: float,
) -> np.ndarray:
    n = len(high)
    out = np.full(n, np.nan, dtype=float)
    if n <= 1:
        return out

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

                sar_value = max(sar_value, prev_high)
                sar_value = max(sar_value, new_high)

                out[out_series_idx] = sar_value
                out_series_idx += 1

                af = acceleration
                ep = new_low

                sar_value = sar_value + (af * (ep - sar_value))

                sar_value = max(sar_value, prev_high)
                sar_value = max(sar_value, new_high)
            else:
                out[out_series_idx] = sar_value
                out_series_idx += 1

                if new_high > ep:
                    ep = new_high
                    af += acceleration
                    af = min(af, maximum)

                sar_value = sar_value + (af * (ep - sar_value))

                sar_value = min(sar_value, prev_low)
                sar_value = min(sar_value, new_low)
        elif new_high >= sar_value:
            is_long = 1
            sar_value = ep

            sar_value = min(sar_value, prev_low)
            sar_value = min(sar_value, new_low)

            out[out_series_idx] = sar_value
            out_series_idx += 1

            af = acceleration
            ep = new_high

            sar_value = sar_value + (af * (ep - sar_value))

            sar_value = min(sar_value, prev_low)
            sar_value = min(sar_value, new_low)
        else:
            out[out_series_idx] = sar_value
            out_series_idx += 1

            if new_low < ep:
                ep = new_low
                af += acceleration
                af = min(af, maximum)

            sar_value = sar_value + (af * (ep - sar_value))

            sar_value = max(sar_value, prev_high)
            sar_value = max(sar_value, new_high)

    return out


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

    if acceleration < 0.0 or acceleration > CMP_N_3E37:
        raise ValueError('sar acceleration must be between 0 and 3e37')
    if maximum < 0.0 or maximum > CMP_N_3E37:
        raise ValueError('sar maximum must be between 0 and 3e37')

    frame = data
    sar_expr = pl.struct([high_col, low_col]).map_batches(
        lambda s: pl.Series(
            _sar_from_arrays(
                s.struct.field(high_col).to_numpy().astype(float, copy=False),
                s.struct.field(low_col).to_numpy().astype(float, copy=False),
                acceleration,
                maximum,
            )
        ),
        return_dtype=pl.Float64,
    ).alias('sar')

    return frame.with_columns(sar_expr)
