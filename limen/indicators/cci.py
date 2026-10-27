import numpy as np
import numpy.typing as npt
import polars as pl
import limen._feature_constraints as _check

CMP_N_100000 = 100000
CMP_N_2 = 2

CCI_SCALING = 0.015


def _cci_from_arrays(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    period: int,
) -> npt.NDArray[np.float64]:
    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    lookback = period - 1
    if n <= lookback:
        return out

    typical_price = (high + low + close) / 3.0
    count = n - lookback
    if count < period:
        prefix = np.zeros(period + 1, dtype=float)
        for start in range(count):
            window = typical_price[start:start + period]
            prefix[1:] = window
            mean = float(np.cumsum(prefix)[-1]) / period
            prefix[1:] = np.abs(window - mean)
            deviation = float(np.cumsum(prefix)[-1])
            delta = window[-1] - mean
            out[start + lookback] = delta / (CCI_SCALING * (deviation / period)) if delta != 0.0 and deviation != 0.0 else 0.0
        return out
    average: npt.NDArray[np.float64] = np.zeros(count, dtype=float)
    for lag in range(period):
        average += typical_price[lag:lag + count]
    _ = np.divide(average, period, out=average)
    abs_sum = np.zeros(count, dtype=float)
    for lag in range(period):
        abs_sum += np.abs(typical_price[lag:lag + count] - average)
    diff = typical_price[lookback:] - average
    _ = np.divide(diff, CCI_SCALING * (abs_sum / period), out=out[lookback:],
              where=(diff != 0.0) & (abs_sum != 0.0))
    out[lookback:][(diff == 0.0) | (abs_sum == 0.0)] = 0.0

    return out


def cci(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    period: int = 14,
) -> pl.DataFrame:

    '''Append CCI with the original chronological mean/deviation sums (period 2..100000).'''

    _check.integers(period=period)
    _check.names(high_col=high_col, low_col=low_col, close_col=close_col)

    if period < CMP_N_2 or period > CMP_N_100000:
        raise ValueError('cci period must be between 2 and 100000')

    out_col = f"cci_{period}"
    frame = data
    cci_expr = pl.struct([high_col, low_col, close_col]).map_batches(
        lambda s: pl.Series(
            _cci_from_arrays(
                s.struct.field(high_col).to_numpy().astype(float, copy=False),
                s.struct.field(low_col).to_numpy().astype(float, copy=False),
                s.struct.field(close_col).to_numpy().astype(float, copy=False),
                period,
            )
        ),
        return_dtype=pl.Float64,
    ).alias(out_col)

    return frame.with_columns(cci_expr)
