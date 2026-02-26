import numpy as np
import polars as pl

from limen.indicators._ema import _ema_talib_default_segment


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

    if fast_period < 2 or fast_period > 100000:
        raise ValueError('fast_period must be between 2 and 100000')
    if slow_period < 2 or slow_period > 100000:
        raise ValueError('slow_period must be between 2 and 100000')
    if signal_period < 1 or signal_period > 100000:
        raise ValueError('signal_period must be between 1 and 100000')

    effective_fast = fast_period
    effective_slow = slow_period
    if effective_slow < effective_fast:
        effective_fast, effective_slow = effective_slow, effective_fast

    values = data[price_col].to_numpy().astype(float, copy=False)
    n = len(values)

    out_macd = np.full(n, np.nan, dtype=float)
    out_signal = np.full(n, np.nan, dtype=float)
    out_hist = np.full(n, np.nan, dtype=float)

    lookback_signal = signal_period - 1
    lookback_total = lookback_signal + (effective_slow - 1)

    if n <= lookback_total:
        return data.with_columns(
            [
                pl.Series(name='macd', values=out_macd),
                pl.Series(name='macd_signal', values=out_signal),
                pl.Series(name='macd_hist', values=out_hist),
            ]
        )

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

    return data.with_columns(
        [
            pl.Series(name='macd', values=out_macd),
            pl.Series(name='macd_signal', values=out_signal),
            pl.Series(name='macd_hist', values=out_hist),
        ]
    )
