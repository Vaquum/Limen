import numpy as np
import numpy.typing as npt


def ema_talib_segment_with_k(
    values: npt.NDArray[np.float64],
    period: int,
    k: float,
    start_idx: int,
    end_idx: int,
) -> tuple[int, npt.NDArray[np.float64]]:
    '''
    TA-Lib INT_EMA equivalent for default compatibility.
    Returns (out_beg_idx, output_values).
    '''
    lookback = period - 1
    start_idx = max(start_idx, lookback)
    if start_idx > end_idx:
        return start_idx, np.empty(0, dtype=float)

    today = start_idx - lookback

    prev_ma = values[today:today + period].mean()
    today += period

    while today <= start_idx:
        prev_ma = ((values[today] - prev_ma) * k) + prev_ma
        today += 1

    out = [prev_ma]
    while today <= end_idx:
        prev_ma = ((values[today] - prev_ma) * k) + prev_ma
        out.append(prev_ma)
        today += 1

    return start_idx, np.asarray(out, dtype=float)


def ema_talib_default_segment(
    values: npt.NDArray[np.float64],
    period: int,
    start_idx: int,
    end_idx: int,
) -> tuple[int, npt.NDArray[np.float64]]:
    k = 2.0 / (period + 1.0)
    return ema_talib_segment_with_k(values, period, k, start_idx, end_idx)
