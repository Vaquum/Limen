import numpy as np
import polars as pl


CMO_EPSILON = 1e-14


def cmo(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 14,
) -> pl.DataFrame:

    '''
    Compute Chande Momentum Oscillator (CMO).

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods (2..100000)

    Returns:
        pl.DataFrame: The input data with a new column 'cmo_{period}'
    '''

    if period < 2 or period > 100000:
        raise ValueError('period must be between 2 and 100000')

    values = data[price_col].to_numpy().astype(float, copy=False)
    n = len(values)
    out_col = f'cmo_{period}'
    out = np.full(n, np.nan, dtype=float)

    lookback_total = period
    if n <= lookback_total:
        return data.with_columns(pl.Series(name=out_col, values=out))

    start_idx = lookback_total
    end_idx = n - 1

    today = start_idx - lookback_total
    prev_value = values[today]

    prev_gain = 0.0
    prev_loss = 0.0
    today += 1
    for _ in range(period):
        current = values[today]
        today += 1
        change = current - prev_value
        prev_value = current
        if change < 0.0:
            prev_loss -= change
        else:
            prev_gain += change

    prev_loss /= period
    prev_gain /= period

    out_idx = start_idx
    denominator = prev_gain + prev_loss
    if abs(denominator) > CMO_EPSILON:
        out[out_idx] = 100.0 * ((prev_gain - prev_loss) / denominator)
    else:
        out[out_idx] = 0.0
    out_idx += 1

    period_minus_one = period - 1.0
    while today <= end_idx:
        current = values[today]
        today += 1
        change = current - prev_value
        prev_value = current

        prev_loss *= period_minus_one
        prev_gain *= period_minus_one
        if change < 0.0:
            prev_loss -= change
        else:
            prev_gain += change

        prev_loss /= period
        prev_gain /= period

        denominator = prev_gain + prev_loss
        if abs(denominator) > CMO_EPSILON:
            out[out_idx] = 100.0 * ((prev_gain - prev_loss) / denominator)
        else:
            out[out_idx] = 0.0
        out_idx += 1

    return data.with_columns(pl.Series(name=out_col, values=out))
