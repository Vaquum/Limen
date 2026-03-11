import numpy as np
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
    raise ValueError('ma_type must be between 0 and 8')


def _stochf_from_arrays(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    fastk_period: int,
    fastd_period: int,
    fastd_ma_type: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    out_fastk = np.full(n, np.nan, dtype=float)
    out_fastd = np.full(n, np.nan, dtype=float)

    lookback_k = fastk_period - 1
    lookback_fastd = _ma_lookback(fastd_period, fastd_ma_type)
    lookback_total = lookback_k + lookback_fastd

    start_idx = lookback_total
    end_idx = n - 1
    if start_idx > end_idx:
        return out_fastk, out_fastd

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
    fastd_col = f"ma_{fastd_period}_{fastd_ma_type}"
    fastd_full = ma(
        fastk_df,
        price_col='_x',
        period=fastd_period,
        ma_type=fastd_ma_type,
    )[fastd_col].to_numpy().astype(float, copy=False)

    fastd_valid = fastd_full[lookback_fastd:]
    fastk_out = fastk_buffer[lookback_fastd:]

    out_count = min(len(fastk_out), len(fastd_valid), n - start_idx)
    if out_count > 0:
        out_fastk[start_idx:start_idx + out_count] = fastk_out[:out_count]
        out_fastd[start_idx:start_idx + out_count] = fastd_valid[:out_count]

    return out_fastk, out_fastd


def _stochf_component(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    fastk_period: int,
    fastd_period: int,
    fastd_ma_type: int,
    component_index: int,
) -> np.ndarray:
    return _stochf_from_arrays(
        high,
        low,
        close,
        fastk_period,
        fastd_period,
        fastd_ma_type,
    )[component_index]


def stochf(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_ma_type: int = 0,
) -> pl.DataFrame:

    '''
    Compute Fast Stochastic Oscillator (TA_STOCHF): fast %K and fast %D.

    Args:
        data (pl.DataFrame): Dataset with high/low/close columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices
        fastk_period (int): Time period for Fast-K (1..100000)
        fastd_period (int): Smoothing period for Fast-D (1..100000)
        fastd_ma_type (int): MA type for Fast-D (0..8)

    Returns:
        pl.DataFrame: The input data with 'stochf_fastk' and 'stochf_fastd'
    '''

    if fastk_period < 1 or fastk_period > CMP_N_100000:
        raise ValueError('fastk_period must be between 1 and 100000')
    if fastd_period < 1 or fastd_period > CMP_N_100000:
        raise ValueError('fastd_period must be between 1 and 100000')
    if fastd_ma_type < 0 or fastd_ma_type > CMP_N_8:
        raise ValueError('fastd_ma_type must be between 0 and 8')

    frame = data
    return frame.with_columns(
        [
            pl.struct([high_col, low_col, close_col]).map_batches(
                lambda s: pl.Series(
                    _stochf_component(
                        s.struct.field(high_col).to_numpy().astype(float, copy=False),
                        s.struct.field(low_col).to_numpy().astype(float, copy=False),
                        s.struct.field(close_col).to_numpy().astype(float, copy=False),
                        fastk_period,
                        fastd_period,
                        fastd_ma_type,
                        0,
                    )
                ),
                return_dtype=pl.Float64,
            ).alias('stochf_fastk'),
            pl.struct([high_col, low_col, close_col]).map_batches(
                lambda s: pl.Series(
                    _stochf_component(
                        s.struct.field(high_col).to_numpy().astype(float, copy=False),
                        s.struct.field(low_col).to_numpy().astype(float, copy=False),
                        s.struct.field(close_col).to_numpy().astype(float, copy=False),
                        fastk_period,
                        fastd_period,
                        fastd_ma_type,
                        1,
                    )
                ),
                return_dtype=pl.Float64,
            ).alias('stochf_fastd'),
        ]
    )
