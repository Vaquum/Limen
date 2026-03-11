import numpy as np
import polars as pl


CMP_N_100000 = 100000
CMP_N_2 = 2

TA_EPSILON = 1e-14


def _rsi_from_values(values: np.ndarray, period: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    if n <= period:
        return out

    prev_value = values[0]
    prev_gain = 0.0
    prev_loss = 0.0

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

    return out


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

    if period < CMP_N_2 or period > CMP_N_100000:
        raise ValueError('period must be between 2 and 100000')

    out_col = f'rsi_{period}'
    frame = data
    rsi_expr = pl.col(price_col).map_batches(
        lambda s: pl.Series(
            _rsi_from_values(
                s.to_numpy().astype(float, copy=False),
                period,
            )
        ),
        return_dtype=pl.Float64,
    ).alias(out_col)

    return frame.with_columns(rsi_expr)
