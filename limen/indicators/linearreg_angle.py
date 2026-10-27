import math
import numpy as np
import numpy.typing as npt
import polars as pl

from limen.indicators.linearreg import linearreg_components as _linearreg_components
import limen._feature_constraints as _check


CMP_N_100000 = 100000
CMP_N_2 = 2

def _linearreg_angle_from_values(values: npt.NDArray[np.float64], period: int) -> npt.NDArray[np.float64]:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)

    lookback_total = period - 1
    if n <= lookback_total:
        return out

    m, _ = _linearreg_components(values, period)
    out[lookback_total:] = np.fromiter((math.atan(value) for value in m), dtype=float) * (180.0 / math.pi)

    return out


def linearreg_angle(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 14,
) -> pl.DataFrame:

    '''
    Compute Linear Regression Angle.

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods

    Returns:
        pl.DataFrame: The input data with a new column 'linearreg_angle_{period}'
    '''

    _check.integers(period=period)
    _check.names(price_col=price_col)

    if period < CMP_N_2 or period > CMP_N_100000:
        raise ValueError('linearreg_angle period must be between 2 and 100000')

    out_col = f"linearreg_angle_{period}"
    return data.with_columns(
        pl.col(price_col).map_batches(
            lambda s: pl.Series(
                _linearreg_angle_from_values(
                    s.to_numpy().astype(float, copy=False),
                    period,
                )
            ),
            return_dtype=pl.Float64,
        ).alias(out_col)
    )
