import numpy as np
import numpy.typing as npt
import polars as pl
import limen._feature_constraints as _check


__all__ = ['midprice', 'window_extrema']

CMP_N_100000 = 100000
CMP_N_2 = 2

def _midprice_from_arrays(high_values: npt.NDArray[np.float64], low_values: npt.NDArray[np.float64], period: int) -> npt.NDArray[np.float64]:
    n = len(high_values)
    out = np.full(n, np.nan, dtype=float)

    lookback_total = period - 1
    if n <= lookback_total:
        return out

    lowest, highest = window_extrema(high_values, low_values, period)
    out[lookback_total:] = (highest + lowest) / 2.0

    return out



def window_extrema(high: npt.NDArray[np.float64], low: npt.NDArray[np.float64],
                   period: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    count = high.size - period + 1
    lowest, highest = low[:count].copy(), high[:count].copy()
    if count < period:
        high_list, low_list = high.tolist(), low.tolist()
        for row in range(count):
            bottom, top = low_list[row], high_list[row]
            for index in range(row + 1, row + period):
                bottom = min(bottom, low_list[index])
                top = max(top, high_list[index])
            lowest[row], highest[row] = bottom, top
    else:
        for lag in range(1, period):
            candidate_low, candidate_high = low[lag:lag + count], high[lag:lag + count]
            lowest = np.where(candidate_low < lowest, candidate_low, lowest)
            highest = np.where(candidate_high > highest, candidate_high, highest)
    return lowest, highest

def midprice(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    period: int = 14,
) -> pl.DataFrame:

    '''Compute the trailing high/low midpoint over period (2..100000).'''

    _check.integers(period=period)
    _check.names(high_col=high_col, low_col=low_col)

    if period < CMP_N_2 or period > CMP_N_100000:
        raise ValueError('midprice period must be between 2 and 100000')

    out_col = f"midprice_{period}"
    return data.with_columns(
        pl.struct([high_col, low_col]).map_batches(
            lambda s: pl.Series(
                _midprice_from_arrays(
                    s.struct.field(high_col).to_numpy().astype(float, copy=False),
                    s.struct.field(low_col).to_numpy().astype(float, copy=False),
                    period,
                )
            ),
            return_dtype=pl.Float64,
        ).alias(out_col)
    )
