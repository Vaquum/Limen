import numpy as np
import polars as pl

from limen.indicators._ema import _ema_talib_default_segment

CMP_N_100000 = 100000
CMP_N_2 = 2

MACD_COL = 'macd'
MACD_SIGNAL_COL = 'macd_signal'
MACD_HIST_COL = 'macd_hist'


def _macd_from_values(
    values: np.ndarray,
    fast_period: int,
    slow_period: int,
    signal_period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(values)
    out_macd = np.full(n, np.nan, dtype=float)
    out_signal = np.full(n, np.nan, dtype=float)
    out_hist = np.full(n, np.nan, dtype=float)

    effective_fast = fast_period
    effective_slow = slow_period
    if effective_slow < effective_fast:
        effective_fast, effective_slow = effective_slow, effective_fast

    lookback_signal = signal_period - 1
    lookback_total = lookback_signal + (effective_slow - 1)
    if n <= lookback_total:
        return out_macd, out_signal, out_hist

    start_idx = lookback_total
    end_idx = n - 1
    ema_start_idx = start_idx - lookback_signal

    _, slow_ema = _ema_talib_default_segment(values, effective_slow, ema_start_idx, end_idx)
    _, fast_ema = _ema_talib_default_segment(values, effective_fast, ema_start_idx, end_idx)
    macd_buffer = fast_ema - slow_ema

    out_count = n - start_idx
    out_macd[start_idx:] = macd_buffer[lookback_signal:lookback_signal + out_count]

    _, signal_values = _ema_talib_default_segment(macd_buffer, signal_period, 0, len(macd_buffer) - 1)
    signal_count = len(signal_values)
    out_signal[start_idx:start_idx + signal_count] = signal_values
    out_hist[start_idx:start_idx + signal_count] = out_macd[start_idx:start_idx + signal_count] - signal_values

    return out_macd, out_signal, out_hist


def _macd_component(
    values: np.ndarray,
    fast_period: int,
    slow_period: int,
    signal_period: int,
    component_index: int,
) -> np.ndarray:
    return _macd_from_values(values, fast_period, slow_period, signal_period)[component_index]


def macd(
    data: pl.DataFrame,
    price_col: str = 'close',
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pl.DataFrame:

    '''
    Compute Moving Average Convergence/Divergence (MACD).

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        fast_period (int): Number of periods for fast EMA (2..100000)
        slow_period (int): Number of periods for slow EMA (2..100000)
        signal_period (int): Number of periods for signal EMA (1..100000)

    Returns:
        pl.DataFrame: The input data with columns 'macd', 'macd_signal', 'macd_hist'
    '''

    if fast_period < CMP_N_2 or fast_period > CMP_N_100000:
        raise ValueError('fast_period must be between 2 and 100000')
    if slow_period < CMP_N_2 or slow_period > CMP_N_100000:
        raise ValueError('slow_period must be between 2 and 100000')
    if signal_period < 1 or signal_period > CMP_N_100000:
        raise ValueError('signal_period must be between 1 and 100000')

    frame = data
    return frame.with_columns(
        [
            pl.col(price_col).map_batches(
                lambda s: pl.Series(
                    _macd_component(
                        s.to_numpy().astype(float, copy=False),
                        fast_period,
                        slow_period,
                        signal_period,
                        0,
                    )
                ),
                return_dtype=pl.Float64,
            ).alias(MACD_COL),
            pl.col(price_col).map_batches(
                lambda s: pl.Series(
                    _macd_component(
                        s.to_numpy().astype(float, copy=False),
                        fast_period,
                        slow_period,
                        signal_period,
                        1,
                    )
                ),
                return_dtype=pl.Float64,
            ).alias(MACD_SIGNAL_COL),
            pl.col(price_col).map_batches(
                lambda s: pl.Series(
                    _macd_component(
                        s.to_numpy().astype(float, copy=False),
                        fast_period,
                        slow_period,
                        signal_period,
                        2,
                    )
                ),
                return_dtype=pl.Float64,
            ).alias(MACD_HIST_COL),
        ]
    )
