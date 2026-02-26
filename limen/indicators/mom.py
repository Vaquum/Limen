import numpy as np
import polars as pl


def mom(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 10,
) -> pl.DataFrame:

    '''
    Compute Momentum (MOM).

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods (1..100000)

    Returns:
        pl.DataFrame: The input data with a new column 'mom_{period}'
    '''

    if period < 1 or period > 100000:
        raise ValueError('period must be between 1 and 100000')

    values = data[price_col].to_numpy().astype(float, copy=False)
    n = len(values)
    out_col = f'mom_{period}'
    out = np.full(n, np.nan, dtype=float)

    if n <= period:
        return data.with_columns(pl.Series(name=out_col, values=out))

    out[period:] = values[period:] - values[:-period]
    return data.with_columns(pl.Series(name=out_col, values=out))
