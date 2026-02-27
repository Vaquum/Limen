import numpy as np
import polars as pl


def rocr100(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 10,
) -> pl.DataFrame:

    '''
    Compute Rate of Change Ratio 100 scale (ROCR100): (price / prev_price) * 100.

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods (1..100000)

    Returns:
        pl.DataFrame: The input data with a new column 'rocr100_{period}'
    '''

    if period < 1 or period > 100000:
        raise ValueError('period must be between 1 and 100000')

    values = data[price_col].to_numpy().astype(float, copy=False)
    n = len(values)
    out_col = f'rocr100_{period}'
    out = np.full(n, np.nan, dtype=float)

    if n <= period:
        return data.with_columns(pl.Series(name=out_col, values=out))

    current = values[period:]
    trailing = values[:-period]
    non_zero_mask = trailing != 0.0

    out_tail = np.zeros(n - period, dtype=float)
    out_tail[non_zero_mask] = (current[non_zero_mask] / trailing[non_zero_mask]) * 100.0
    out[period:] = out_tail

    return data.with_columns(pl.Series(name=out_col, values=out))
