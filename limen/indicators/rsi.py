import numpy as np
import polars as pl

TA_EPSILON = 1e-14


def rsi(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 14,
) -> pl.DataFrame:

    '''
    Compute Relative Strength Index (RSI) using Wilder smoothing.

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods (2..100000)

    Returns:
        pl.DataFrame: The input data with a new column 'rsi_{period}'
    '''

    if period < 2 or period > 100000:
        raise ValueError('period must be between 2 and 100000')

    values = data[price_col].to_numpy().astype(float, copy=False)
    n = len(values)
    out_col = f'rsi_{period}'
    out = np.full(n, np.nan, dtype=float)

    if n <= period:
        return data.with_columns(pl.Series(name=out_col, values=out))

    prev_value = values[0]
    prev_gain = 0.0
    prev_loss = 0.0

    # Initial Wilder average gain/loss over the first period deltas.
    for today in range(1, period + 1):
        diff = values[today] - prev_value
        prev_value = values[today]
        if diff < 0.0:
            prev_loss -= diff
        else:
            prev_gain += diff

    prev_loss /= period
    prev_gain /= period

    denom = prev_gain + prev_loss
    out[period] = 100.0 * (prev_gain / denom) if abs(denom) >= TA_EPSILON else 0.0

    # Wilder smoothing for subsequent points.
    for today in range(period + 1, n):
        diff = values[today] - prev_value
        prev_value = values[today]

        prev_loss *= (period - 1)
        prev_gain *= (period - 1)

        if diff < 0.0:
            prev_loss -= diff
        else:
            prev_gain += diff

        prev_loss /= period
        prev_gain /= period

        denom = prev_gain + prev_loss
        out[today] = 100.0 * (prev_gain / denom) if abs(denom) >= TA_EPSILON else 0.0

    return data.with_columns(pl.Series(name=out_col, values=out))
