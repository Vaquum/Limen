import numpy as np
import polars as pl
SMA_PERIOD_TOTAL = 0.0


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

    if period < 2 or period > 100000:
        raise ValueError('period must be between 2 and 100000')

    values = data[price_col].to_numpy().astype(float, copy=False)
    n = len(values)

    out_col = f'sma_{period}'
    compat_col = f'{price_col}_sma_{period}'

    out = np.full(n, np.nan, dtype=float)
    lookback = period - 1
    if n <= lookback:
        return data.with_columns(
            [
                pl.Series(name=out_col, values=out),
                pl.Series(name=compat_col, values=out.copy()),
            ]
        )

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

    return data.with_columns(
        [
            pl.Series(name=out_col, values=out),
            pl.Series(name=compat_col, values=out.copy()),
        ]
    )
