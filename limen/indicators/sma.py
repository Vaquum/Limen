import numpy as np
import polars as pl
CMP_N_100000 = 100000
CMP_N_2 = 2

SMA_PERIOD_TOTAL = 0.0


def _sma_from_values(values: np.ndarray, period: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    lookback = period - 1
    if n <= lookback:
        return out

    period_total = SMA_PERIOD_TOTAL
    trailing_idx = 0
    i = 0
    while i < lookback:
        period_total += values[i]
        i += 1

    while i < n:
        period_total += values[i]
        temp_real = period_total
        period_total -= values[trailing_idx]
        trailing_idx += 1
        out[i] = temp_real / period
        i += 1

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
        price_col = column

    if period < CMP_N_2 or period > CMP_N_100000:
        raise ValueError('sma period must be between 2 and 100000')

    out_col = f"sma_{period}"
    compat_col = f"{price_col}_sma_{period}"
    frame = data
    sma_expr = pl.col(price_col).map_batches(
        lambda s: pl.Series(
            _sma_from_values(
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
