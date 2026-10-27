import numpy as np
import numpy.typing as npt
import polars as pl
import limen._feature_constraints as _check
CMP_N_100000 = 100000
CMP_N_2 = 2

SMA_PERIOD_TOTAL = 0.0


def sma_from_values(values: npt.NDArray[np.float64], period: int) -> npt.NDArray[np.float64]:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    lookback = period - 1
    if n <= lookback:
        return out

    period_total = SMA_PERIOD_TOTAL
    i = 0
    while i < lookback:
        period_total += values[i]
        i += 1

    count = n - i
    steps = np.empty(2 * count + 1)
    steps[0] = period_total
    steps[1::2] = values[i:]
    steps[2::2] = -values[:count]
    steps = np.cumsum(steps, out=steps)
    out[i:] = steps[1::2] / period

    return out


def sma(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 30,
    column: str | None = None,
) -> pl.DataFrame:

    '''
    Compute Simple Moving Average (SMA).

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods
        column (str | None): Backward-compatible alias for price_col

    Returns:
        pl.DataFrame: Input data with a new column 'sma_{period}'.
            Also includes '{price_col}_sma_{period}' as a compatibility alias.
    '''

    if column is not None:
        _check.names(column=column)
    _check.integers(period=period)
    _check.names(price_col=price_col)

    if column is not None:
        price_col = column

    if period < CMP_N_2 or period > CMP_N_100000:
        raise ValueError('sma period must be between 2 and 100000')

    out_col = f"sma_{period}"
    compat_col = f"{price_col}_sma_{period}"
    frame = data
    sma_expr = pl.col(price_col).map_batches(
        lambda s: pl.Series(
            sma_from_values(
                s.to_numpy().astype(float, copy=False),
                period,
            )
        ),
        return_dtype=pl.Float64,
    ).alias(out_col)

    return frame.with_columns(sma_expr).with_columns(
        [
            pl.col(out_col).alias(compat_col),
        ]
    )
