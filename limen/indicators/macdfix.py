import numpy as np
import polars as pl

from limen.indicators._ema import _ema_talib_default_segment, _ema_talib_segment_with_k

CMP_N_100000 = 100000

MACDFIX_FIXED_FAST_PERIOD = 12
MACDFIX_FIXED_SLOW_PERIOD = 26
MACDFIX_COL = 'macdfix'
MACDFIX_SIGNAL_COL = 'macdfix_signal'
MACDFIX_HIST_COL = 'macdfix_hist'


def _macdfix_from_values(
    values: np.ndarray,
    signal_period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(values)
    out_macd = np.full(n, np.nan, dtype=float)
    out_signal = np.full(n, np.nan, dtype=float)
    out_hist = np.full(n, np.nan, dtype=float)

    fixed_fast_period = MACDFIX_FIXED_FAST_PERIOD
    fixed_slow_period = MACDFIX_FIXED_SLOW_PERIOD
    fixed_fast_k = 0.15
    fixed_slow_k = 0.075

    lookback_signal = signal_period - 1
    lookback_total = lookback_signal + (fixed_slow_period - 1)
    if n <= lookback_total:
        return out_macd, out_signal, out_hist

    start_idx = lookback_total
    end_idx = n - 1
    ema_start_idx = start_idx - lookback_signal

    _, slow_ema = _ema_talib_segment_with_k(
        values,
        fixed_slow_period,
        fixed_slow_k,
        ema_start_idx,
        end_idx,
    )
    _, fast_ema = _ema_talib_segment_with_k(
        values,
        fixed_fast_period,
        fixed_fast_k,
        ema_start_idx,
        end_idx,
    )
    macd_buffer = fast_ema - slow_ema

    out_count = n - start_idx
    out_macd[start_idx:] = macd_buffer[lookback_signal:lookback_signal + out_count]

    _, signal_values = _ema_talib_default_segment(macd_buffer, signal_period, 0, len(macd_buffer) - 1)
    signal_count = len(signal_values)
    out_signal[start_idx:start_idx + signal_count] = signal_values
    out_hist[start_idx:start_idx + signal_count] = out_macd[start_idx:start_idx + signal_count] - signal_values

    return out_macd, out_signal, out_hist


def macdfix(
    data: pl.DataFrame,
    price_col: str = 'close',
    signal_period: int = 9,
) -> pl.DataFrame:

    '''
    Compute MACD Fix 12/26 (MACDFIX).

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        signal_period (int): Number of periods for signal EMA (1..100000)

    Returns:
        pl.DataFrame: The input data with columns 'macdfix', 'macdfix_signal', 'macdfix_hist'
    '''

    if signal_period < 1 or signal_period > CMP_N_100000:
        raise ValueError('signal_period must be between 1 and 100000')

    values = data[price_col].to_numpy().astype(float, copy=False)
    macd_values, signal_values, hist_values = _macdfix_from_values(values, signal_period)

    return data.with_columns(
        [
            pl.Series(name=MACDFIX_COL, values=macd_values),
            pl.Series(name=MACDFIX_SIGNAL_COL, values=signal_values),
            pl.Series(name=MACDFIX_HIST_COL, values=hist_values),
        ]
    )
