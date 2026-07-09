import numpy as np
import numpy.typing as npt
import polars as pl

from limen.indicators.ma import ma


CMP_N_100000 = 100000
CMP_N_3 = 3
CMP_N_4 = 4
CMP_N_6 = 6
CMP_N_7 = 7
CMP_N_8 = 8

def _ma_lookback(period: int, ma_type: int) -> int:
    if period <= 1:
        return 0
    lookback_by_type = {
        0: period - 1,
        1: period - 1,
        2: period - 1,
        5: period - 1,
        CMP_N_3: 2 * (period - 1),
        CMP_N_4: 3 * (period - 1),
        CMP_N_6: period,
        CMP_N_7: 32,
        CMP_N_8: 6 * (period - 1),
    }
    if ma_type in lookback_by_type:
        return lookback_by_type[ma_type]
    raise ValueError('stoch ma_type must be between 0 and 8')


def _stoch_from_arrays(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    fastk_period: int,
    slowk_period: int,
    slowk_ma_type: int,
    slowd_period: int,
    slowd_ma_type: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    n = len(close)
    out_slowk = np.full(n, np.nan, dtype=float)
    out_slowd = np.full(n, np.nan, dtype=float)

    lookback_k = fastk_period - 1
    lookback_k_slow = _ma_lookback(slowk_period, slowk_ma_type)
    lookback_d_slow = _ma_lookback(slowd_period, slowd_ma_type)
    lookback_total = lookback_k + lookback_k_slow + lookback_d_slow

    start_idx = lookback_total
    end_idx = n - 1
    if start_idx > end_idx:
        return out_slowk, out_slowd

    trailing_idx = start_idx - lookback_total
    today = trailing_idx + lookback_k
    lowest_idx = -1
    highest_idx = -1
    lowest = 0.0
    highest = 0.0
    diff = 0.0

    fastk_buffer = np.empty(end_idx - today + 1, dtype=float)
    out_idx = 0

    while today <= end_idx:
        tmp = low[today]
        if lowest_idx < trailing_idx:
            lowest_idx = trailing_idx
            lowest = low[lowest_idx]
            i = lowest_idx + 1
            while i <= today:
                tmp = low[i]
                if tmp < lowest:
                    lowest_idx = i
                    lowest = tmp
                i += 1
            diff = (highest - lowest) / 100.0
        elif tmp <= lowest:
            lowest_idx = today
            lowest = tmp
            diff = (highest - lowest) / 100.0

        tmp = high[today]
        if highest_idx < trailing_idx:
            highest_idx = trailing_idx
            highest = high[highest_idx]
            i = highest_idx + 1
            while i <= today:
                tmp = high[i]
                if tmp > highest:
                    highest_idx = i
                    highest = tmp
                i += 1
            diff = (highest - lowest) / 100.0
        elif tmp >= highest:
            highest_idx = today
            highest = tmp
            diff = (highest - lowest) / 100.0

        if diff != 0.0:
            fastk_buffer[out_idx] = (close[today] - lowest) / diff
        else:
            fastk_buffer[out_idx] = 0.0

        out_idx += 1
        trailing_idx += 1
        today += 1

    fastk_df = pl.DataFrame({'_x': fastk_buffer})
    slowk_col = f"ma_{slowk_period}_{slowk_ma_type}"
    slowk_full = ma(
        fastk_df,
        price_col='_x',
        period=slowk_period,
        ma_type=slowk_ma_type,
    )[slowk_col].to_numpy().astype(float, copy=False)

    slowk_valid = slowk_full[lookback_k_slow:]

    slowd_df = pl.DataFrame({'_x': slowk_valid})
    slowd_col = f"ma_{slowd_period}_{slowd_ma_type}"
    slowd_full = ma(
        slowd_df,
        price_col='_x',
        period=slowd_period,
        ma_type=slowd_ma_type,
    )[slowd_col].to_numpy().astype(float, copy=False)

    slowd_valid = slowd_full[lookback_d_slow:]
    slowk_out = slowk_valid[lookback_d_slow:]

    out_count = min(len(slowk_out), len(slowd_valid), n - start_idx)
    if out_count > 0:
        out_slowk[start_idx:start_idx + out_count] = slowk_out[:out_count]
        out_slowd[start_idx:start_idx + out_count] = slowd_valid[:out_count]

    return out_slowk, out_slowd


def stoch(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    fastk_period: int = 5,
    slowk_period: int = 3,
    slowk_ma_type: int = 0,
    slowd_period: int = 3,
    slowd_ma_type: int = 0,
) -> pl.DataFrame:

    '''
    Compute Stochastic Oscillator (TA_STOCH): slow %K and slow %D.

    Args:
        data (pl.DataFrame): Dataset with high/low/close columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices
        fastk_period (int): Time period for Fast-K (1..100000)
        slowk_period (int): Smoothing period for Slow-K (1..100000)
        slowk_ma_type (int): MA type for Slow-K (0..8)
        slowd_period (int): Smoothing period for Slow-D (1..100000)
        slowd_ma_type (int): MA type for Slow-D (0..8)

    Returns:
        pl.DataFrame: The input data with 'stoch_slowk' and 'stoch_slowd'
    '''

    if fastk_period < 1 or fastk_period > CMP_N_100000:
        raise ValueError('stoch fastk_period must be between 1 and 100000')
    if slowk_period < 1 or slowk_period > CMP_N_100000:
        raise ValueError('stoch slowk_period must be between 1 and 100000')
    if slowd_period < 1 or slowd_period > CMP_N_100000:
        raise ValueError('stoch slowd_period must be between 1 and 100000')
    if slowk_ma_type < 0 or slowk_ma_type > CMP_N_8:
        raise ValueError('stoch slowk_ma_type must be between 0 and 8')
    if slowd_ma_type < 0 or slowd_ma_type > CMP_N_8:
        raise ValueError('stoch slowd_ma_type must be between 0 and 8')

    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    slowk_values, slowd_values = _stoch_from_arrays(
        high_values,
        low_values,
        close_values,
        fastk_period,
        slowk_period,
        slowk_ma_type,
        slowd_period,
        slowd_ma_type,
    )

    return data.with_columns(
        [
            pl.Series(name='stoch_slowk', values=slowk_values),
            pl.Series(name='stoch_slowd', values=slowd_values),
        ]
    )
