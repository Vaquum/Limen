import numpy as np
import numpy.typing as npt
import polars as pl

from limen.indicators.ma import ma


CMP_N_100000 = 100000
CMP_N_2 = 2
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
    raise ValueError('macdext ma_type must be between 0 and 8')


def _ma_talib_segment(
    values: npt.NDArray[np.float64],
    period: int,
    ma_type: int,
    start_idx: int,
    end_idx: int,
) -> npt.NDArray[np.float64]:
    lookback = _ma_lookback(period, ma_type)
    start = max(start_idx, lookback)
    if start > end_idx:
        return np.empty(0, dtype=float)

    base_idx = start - lookback
    segment = values[base_idx:end_idx + 1]
    segment_df = pl.DataFrame({'_x': segment})
    ma_col = f"ma_{period}_{ma_type}"
    segment_ma = ma(
        segment_df,
        price_col='_x',
        period=period,
        ma_type=ma_type,
    )[ma_col].to_numpy().astype(float, copy=False)

    return segment_ma[lookback:]


def _macdext_from_values(
    values: npt.NDArray[np.float64],
    fast_period: int,
    fast_ma_type: int,
    slow_period: int,
    slow_ma_type: int,
    signal_period: int,
    signal_ma_type: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    n = len(values)
    out_macd = np.full(n, np.nan, dtype=float)
    out_signal = np.full(n, np.nan, dtype=float)
    out_hist = np.full(n, np.nan, dtype=float)

    effective_fast_period = fast_period
    effective_fast_type = fast_ma_type
    effective_slow_period = slow_period
    effective_slow_type = slow_ma_type
    if effective_slow_period < effective_fast_period:
        effective_fast_period, effective_slow_period = effective_slow_period, effective_fast_period
        effective_fast_type, effective_slow_type = effective_slow_type, effective_fast_type

    lookback_largest = max(
        _ma_lookback(effective_fast_period, effective_fast_type),
        _ma_lookback(effective_slow_period, effective_slow_type),
    )
    lookback_signal = _ma_lookback(signal_period, signal_ma_type)
    lookback_total = lookback_largest + lookback_signal
    if n <= lookback_total:
        return out_macd, out_signal, out_hist

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
    return out_macd, out_signal, out_hist


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

    if fast_period < CMP_N_2 or fast_period > CMP_N_100000:
        raise ValueError('macdext fast_period must be between 2 and 100000')
    if slow_period < CMP_N_2 or slow_period > CMP_N_100000:
        raise ValueError('macdext slow_period must be between 2 and 100000')
    if signal_period < 1 or signal_period > CMP_N_100000:
        raise ValueError('macdext signal_period must be between 1 and 100000')
    if fast_ma_type < 0 or fast_ma_type > CMP_N_8:
        raise ValueError('macdext fast_ma_type must be between 0 and 8')
    if slow_ma_type < 0 or slow_ma_type > CMP_N_8:
        raise ValueError('macdext slow_ma_type must be between 0 and 8')
    if signal_ma_type < 0 or signal_ma_type > CMP_N_8:
        raise ValueError('macdext signal_ma_type must be between 0 and 8')

    def _macdext_struct(series: pl.Series) -> pl.Series:
        values = series.to_numpy().astype(float, copy=False)
        macd_values, signal_values, hist_values = _macdext_from_values(
            values,
            fast_period,
            fast_ma_type,
            slow_period,
            slow_ma_type,
            signal_period,
            signal_ma_type,
        )
        return pl.DataFrame({
            'macdext': macd_values,
            'macdext_signal': signal_values,
            'macdext_hist': hist_values,
        }).to_struct('__macdext_struct')

    macdext_expr = pl.col(price_col).map_batches(
        _macdext_struct,
        return_dtype=pl.Struct({
            'macdext': pl.Float64,
            'macdext_signal': pl.Float64,
            'macdext_hist': pl.Float64,
        }),
    ).alias('__macdext_struct')

    return data.with_columns(macdext_expr).unnest('__macdext_struct')
