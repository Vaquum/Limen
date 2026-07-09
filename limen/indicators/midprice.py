import numpy as np
import numpy.typing as npt
import polars as pl


CMP_N_100000 = 100000
CMP_N_2 = 2

def _midprice_from_arrays(high_values: npt.NDArray[np.float64], low_values: npt.NDArray[np.float64], period: int) -> npt.NDArray[np.float64]:
    n = len(high_values)
    out = np.full(n, np.nan, dtype=float)

    lookback_total = period - 1
    if n <= lookback_total:
        return out

    today = lookback_total
    trailing_idx = 0
    while today < n:
        lowest = low_values[trailing_idx]
        highest = high_values[trailing_idx]
        i = trailing_idx + 1
        while i <= today:
            low_tmp = low_values[i]
            lowest = min(lowest, low_tmp)
            high_tmp = high_values[i]
            highest = max(highest, high_tmp)
            i += 1

        out[today] = (highest + lowest) / 2.0
        today += 1
        trailing_idx += 1

    return out


def midprice(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    period: int = 14,
) -> pl.DataFrame:

    '''
    Compute Midpoint Price over period.

    Args:
        data (pl.DataFrame): Dataset with high and low columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        period (int): Number of periods (2..100000)

    Returns:
        pl.DataFrame: The input data with a new column 'midprice_{period}'
    '''

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
