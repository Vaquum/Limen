import polars as pl
import numpy as np

def random_slice(
    df: pl.DataFrame,
    rows: int,
    *,
    seed: int | None = None
) -> pl.DataFrame:
    '''
    Grab a contiguous slice rows long.
    Start index is uniform in [25 %, 75 %] of the frame (inclusive) and
    guaranteed to fit the window length.

    Args:
        df (pl.DataFrame): Input DataFrame to slice from
        rows (int): Number of rows to include in the slice
        seed (int | None, optional): Random seed for reproducible results. 
            Defaults to None for non-deterministic behavior.
    
    Returns:
        pl.DataFrame: A contiguous slice of the original DataFrame with 'rows' number of rows,
            maintaining the original order of rows.
    
    Raises:
        ValueError: If the requested slice size is too large to fit within the 
            25%-75% range of the DataFrame.
    '''
    n = len(df)
    lo = int(n * 0.25)
    hi = int(n * 0.75) - rows  # highest valid start
    if hi < lo:
        raise ValueError('slice size too large for chosen 25–75 % range')

    rng = np.random.default_rng(seed)
    start = int(rng.integers(lo, hi + 1))
    return df[start : start + rows]  # slice keeps order
