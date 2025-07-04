import polars as pl
import numpy as np

def random_slice(
    df: pl.DataFrame,
    rows: int,
    *,
    safe_range_low: float = 0.25,
    safe_range_high: float = 0.75,
    seed: int | None = None
) -> pl.DataFrame:
    '''
    Grab a contiguous slice rows long.
    Start index is uniform in [safe_range_low, safe_range_high] of the frame (inclusive) and
    guaranteed to fit the window length.

    Args:
        df (pl.DataFrame): Input DataFrame to slice from
        rows (int): Number of rows to include in the slice
        safe_range_low (float, optional): Lower bound of safe range as fraction of total rows (0.0 to 1.0).
            Defaults to 0.25 (25%).
        safe_range_high (float, optional): Upper bound of safe range as fraction of total rows (0.0 to 1.0).
            Defaults to 0.75 (75%).
        seed (int | None, optional): Random seed for reproducible results. 
            Defaults to None for non-deterministic behavior.
    
    Returns:
        pl.DataFrame: A contiguous slice of the original DataFrame with 'rows' number of rows,
            maintaining the original order of rows.
    
    Raises:
        ValueError: If the requested slice size is too large to fit within the 
            specified safe range, or if safe range parameters are invalid.
    '''
    # Validate safe range parameters
    if not (0.0 <= safe_range_low < safe_range_high <= 1.0):
        raise ValueError('safe_range_low must be >= 0.0, safe_range_high must be <= 1.0, and safe_range_low < safe_range_high')

    n = len(df)
    lo = int(n * safe_range_low)
    hi = int(n * safe_range_high) - rows  # highest valid start

    if hi < lo:
        raise ValueError(f'slice size ({rows}) too large for chosen safe range ({safe_range_low*100:.0f}%-{safe_range_high*100:.0f}%)')

    rng = np.random.default_rng(seed)
    start = int(rng.integers(lo, hi + 1))
    return df[start : start + rows]  # slice keeps order