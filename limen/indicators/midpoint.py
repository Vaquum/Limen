import numpy as np
import polars as pl


def midpoint(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 14,
) -> pl.DataFrame:

    '''
    Compute MidPoint over period.

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods (2..100000)

    Returns:
        pl.DataFrame: The input data with a new column 'midpoint_{period}'
    '''

    if period < 2 or period > 100000:
        raise ValueError('period must be between 2 and 100000')

    values = data[price_col].to_numpy().astype(float, copy=False)
    n = len(values)
    out_col = f'midpoint_{period}'
    out = np.full(n, np.nan, dtype=float)

    lookback_total = period - 1
    if n <= lookback_total:
        return data.with_columns(pl.Series(name=out_col, values=out))

    today = lookback_total
    trailing_idx = 0
    while today < n:
        lowest = values[trailing_idx]
        highest = lowest
        i = trailing_idx + 1
        while i <= today:
            tmp = values[i]
            if tmp < lowest:
                lowest = tmp
            elif tmp > highest:
                highest = tmp
            i += 1

        out[today] = (highest + lowest) / 2.0
        today += 1
        trailing_idx += 1

    return data.with_columns(pl.Series(name=out_col, values=out))
