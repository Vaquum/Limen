import numpy as np
import polars as pl


BOP_EPSILON = 1e-14


def bop(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Balance of Power (BOP).

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'bop'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)

    denominator = high_values - low_values
    out = np.zeros(len(data), dtype=float)

    valid = denominator >= BOP_EPSILON
    out[valid] = (close_values[valid] - open_values[valid]) / denominator[valid]

    return data.with_columns(pl.Series(name='bop', values=out))
