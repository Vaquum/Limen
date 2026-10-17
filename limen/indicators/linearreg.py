import numpy as np
import numpy.typing as npt
import polars as pl


CMP_N_100000 = 100000
CMP_N_2 = 2

def _linearreg_from_values(values: npt.NDArray[np.float64], period: int) -> npt.NDArray[np.float64]:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)

    lookback_total = period - 1
    if n <= lookback_total:
        return out

    m, b = _linearreg_components(values, period)
    out[lookback_total:] = b + (m * float(period - 1))

    return out



def _linearreg_components(
    values: npt.NDArray[np.float64], period: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    count = len(values) - period + 1
    sum_y = np.zeros(count)
    sum_xy = np.zeros(count)
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

    '''
    Compute Linear Regression.

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods

    Returns:
        pl.DataFrame: The input data with a new column 'linearreg_{period}'
    '''

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
