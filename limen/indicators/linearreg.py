import numpy as np
import numpy.typing as npt
import polars as pl
import limen._feature_constraints as _check

__all__ = ['linearreg', 'linearreg_components']


CMP_N_100000 = 100000
CMP_N_2 = 2

def _linearreg_from_values(values: npt.NDArray[np.float64], period: int) -> npt.NDArray[np.float64]:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)

    lookback_total = period - 1
    if n <= lookback_total:
        return out

    m, b = linearreg_components(values, period)
    out[lookback_total:] = b + (m * float(period - 1))

    return out



def linearreg_components(
    values: npt.NDArray[np.float64], period: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    count = len(values) - period + 1
    sum_y = np.zeros(count)
    sum_xy = np.zeros(count)
    if count < period:
        prefix = np.zeros(period + 1, dtype=float)
        weights = np.arange(period - 1, -1, -1, dtype=float)
        for start in range(count):
            window = values[start:start + period]
            prefix[1:] = window
            sum_y[start] = np.cumsum(prefix)[-1]
            prefix[1:] = weights * window
            sum_xy[start] = np.cumsum(prefix)[-1]
    else:
        for lag in range(period):
            window = values[lag:lag + count]
            sum_y += window
            sum_xy += float(period - 1 - lag) * window
    sum_x = period * (period - 1) * 0.5
    sum_x_sqr = period * (period - 1) * (2 * period - 1) / 6.0
    divisor = (sum_x * sum_x) - (period * sum_x_sqr)
    m = ((period * sum_xy) - (sum_x * sum_y)) / divisor
    return m, (sum_y - (m * sum_x)) / float(period)

def linearreg(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 14,
) -> pl.DataFrame:

    '''Append the trailing least-squares line value with the original warmup.'''

    _check.integers(period=period)
    _check.names(price_col=price_col)

    if period < CMP_N_2 or period > CMP_N_100000:
        raise ValueError('linearreg period must be between 2 and 100000')

    out_col = f"linearreg_{period}"
    return data.with_columns(
        pl.col(price_col).map_batches(
            lambda s: pl.Series(
                _linearreg_from_values(
                    s.to_numpy().astype(float, copy=False),
                    period,
                )
            ),
            return_dtype=pl.Float64,
        ).alias(out_col)
    )
