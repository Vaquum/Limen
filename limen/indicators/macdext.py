import numpy as np
import polars as pl

from limen.indicators.ma import ma


def _ma_lookback(period: int, ma_type: int) -> int:
    if period <= 1:
        return 0
    if ma_type in (0, 1, 2, 5):
        return period - 1
    if ma_type == 3:
        return 2 * (period - 1)
    if ma_type == 4:
        return 3 * (period - 1)
    if ma_type == 6:
        return period
    if ma_type == 7:
        return 32
    if ma_type == 8:
        return 6 * (period - 1)
    raise ValueError('ma_type must be between 0 and 8')


def _ma_talib_segment(
    values: np.ndarray,
    period: int,
    ma_type: int,
    start_idx: int,
    end_idx: int,
) -> np.ndarray:
    lookback = _ma_lookback(period, ma_type)
    start = max(start_idx, lookback)
    if start > end_idx:
        return np.empty(0, dtype=float)

    base_idx = start - lookback
    segment = values[base_idx:end_idx + 1]
    segment_df = pl.DataFrame({'_x': segment})
    ma_col = f'ma_{period}_{ma_type}'
    segment_ma = ma(
        segment_df,
        price_col='_x',
        period=period,
        ma_type=ma_type,
    )[ma_col].to_numpy().astype(float, copy=False)

    return segment_ma[lookback:]


def macdext(
    data: pl.DataFrame,
    price_col: str = 'close',
    fast_period: int = 12,
    fast_ma_type: int = 0,
    slow_period: int = 26,
    slow_ma_type: int = 0,
    signal_period: int = 9,
    signal_ma_type: int = 0,
) -> pl.DataFrame:

    '''
    Compute MACD with controllable MA types (MACDEXT).

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        fast_period (int): Number of periods for fast MA (2..100000)
        fast_ma_type (int): TA-Lib MA type for fast MA (0..8)
        slow_period (int): Number of periods for slow MA (2..100000)
        slow_ma_type (int): TA-Lib MA type for slow MA (0..8)
        signal_period (int): Number of periods for signal MA (1..100000)
        signal_ma_type (int): TA-Lib MA type for signal MA (0..8)

    Returns:
        pl.DataFrame: The input data with columns 'macdext', 'macdext_signal', 'macdext_hist'
    '''

    if fast_period < 2 or fast_period > 100000:
        raise ValueError('fast_period must be between 2 and 100000')
    if slow_period < 2 or slow_period > 100000:
        raise ValueError('slow_period must be between 2 and 100000')
    if signal_period < 1 or signal_period > 100000:
        raise ValueError('signal_period must be between 1 and 100000')
    if fast_ma_type < 0 or fast_ma_type > 8:
        raise ValueError('fast_ma_type must be between 0 and 8')
    if slow_ma_type < 0 or slow_ma_type > 8:
        raise ValueError('slow_ma_type must be between 0 and 8')
    if signal_ma_type < 0 or signal_ma_type > 8:
        raise ValueError('signal_ma_type must be between 0 and 8')

    effective_fast_period = fast_period
    effective_fast_type = fast_ma_type
    effective_slow_period = slow_period
    effective_slow_type = slow_ma_type
    if effective_slow_period < effective_fast_period:
        effective_fast_period, effective_slow_period = effective_slow_period, effective_fast_period
        effective_fast_type, effective_slow_type = effective_slow_type, effective_fast_type

    values = data[price_col].to_numpy().astype(float, copy=False)
    n = len(values)

    out_macd = np.full(n, np.nan, dtype=float)
    out_signal = np.full(n, np.nan, dtype=float)
    out_hist = np.full(n, np.nan, dtype=float)

    lookback_largest = max(
        _ma_lookback(effective_fast_period, effective_fast_type),
        _ma_lookback(effective_slow_period, effective_slow_type),
    )
    lookback_signal = _ma_lookback(signal_period, signal_ma_type)
    lookback_total = lookback_largest + lookback_signal

    if n <= lookback_total:
        return data.with_columns(
            [
                pl.Series(name='macdext', values=out_macd),
                pl.Series(name='macdext_signal', values=out_signal),
                pl.Series(name='macdext_hist', values=out_hist),
            ]
        )

    start_idx = lookback_total
    ma_start_idx = start_idx - lookback_signal
    end_idx = n - 1

    fast_ma_buffer = _ma_talib_segment(
        values,
        effective_fast_period,
        effective_fast_type,
        ma_start_idx,
        end_idx,
    )
    slow_ma_buffer = _ma_talib_segment(
        values,
        effective_slow_period,
        effective_slow_type,
        ma_start_idx,
        end_idx,
    )

    macd_buffer = fast_ma_buffer - slow_ma_buffer
    out_count = n - start_idx
    out_macd[start_idx:] = macd_buffer[lookback_signal:lookback_signal + out_count]

    signal_values = _ma_talib_segment(
        macd_buffer,
        signal_period,
        signal_ma_type,
        0,
        len(macd_buffer) - 1,
    )

    out_signal[start_idx:] = signal_values[:out_count]
    out_hist[start_idx:] = out_macd[start_idx:] - out_signal[start_idx:]

    return data.with_columns(
        [
            pl.Series(name='macdext', values=out_macd),
            pl.Series(name='macdext_signal', values=out_signal),
            pl.Series(name='macdext_hist', values=out_hist),
        ]
    )
