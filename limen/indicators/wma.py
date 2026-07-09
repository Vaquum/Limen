import numpy as np
import numpy.typing as npt
import polars as pl

CMP_N_100000 = 100000
CMP_N_2 = 2

WMA_PERIOD_SUB = 0.0
WMA_PERIOD_SUM = 0.0


def wma_from_values(values: npt.NDArray[np.float64], period: int) -> npt.NDArray[np.float64]:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)

    lookback_total = period - 1
    if n <= lookback_total:
        return out

    start_idx = lookback_total
    end_idx = n - 1

    divider = (period * (period + 1)) // 2

    out_idx = start_idx
    trailing_idx = start_idx - lookback_total

    period_sum = WMA_PERIOD_SUM
    period_sub = WMA_PERIOD_SUB
    in_idx = trailing_idx
    i = 1
    while in_idx < start_idx:
        temp_real = values[in_idx]
        in_idx += 1
        period_sub += temp_real
        period_sum += temp_real * i
        i += 1

    trailing_value = 0.0

    while in_idx <= end_idx:
        temp_real = values[in_idx]
        in_idx += 1

        period_sub += temp_real
        period_sub -= trailing_value
        period_sum += temp_real * period

        trailing_value = values[trailing_idx]
        trailing_idx += 1

        out[out_idx] = period_sum / divider
        out_idx += 1

        period_sum -= period_sub

    return out


def wma(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 30,
) -> pl.DataFrame:

    '''
    Compute Weighted Moving Average (WMA).

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods

    Returns:
        pl.DataFrame: The input data with a new column 'wma_{period}'
    '''

    if period < CMP_N_2 or period > CMP_N_100000:
        raise ValueError('wma period must be between 2 and 100000')

    out_col = f"wma_{period}"
    frame = data
    wma_expr = pl.col(price_col).map_batches(
        lambda s: pl.Series(
            wma_from_values(
                s.to_numpy().astype(float, copy=False),
                period,
            )
        ),
        return_dtype=pl.Float64,
    ).alias(out_col)

    return frame.with_columns(wma_expr)
