import polars as pl
import numpy as np
from typing import List


DEFAULT_SAMPLE_SIZE = 10000
DEFAULT_N_SAMPLES = 3


def random_subsets_sampling(data: pl.DataFrame, 
                           sample_size: int = DEFAULT_SAMPLE_SIZE,
                           n_samples: int = DEFAULT_N_SAMPLES,
                           *,
                           safe_range_low: float = 0.25,
                           safe_range_high: float = 0.75,
                           seed: int | None = None) -> List[pl.DataFrame]:

    '''
    Compute random contiguous subsets from dataset avoiding edge effects.

    Args:
        data (pl.DataFrame): Klines dataset with 'datetime' and numeric columns
        sample_size (int): Number of rows in each contiguous subset
        n_samples (int): Number of random subsets to create
        safe_range_low (float): Lower bound of safe range as fraction of total rows
        safe_range_high (float): Upper bound of safe range as fraction of total rows
        seed (int | None): Random seed for reproducible results
        
    Returns:
        List[pl.DataFrame]: List of random contiguous subsets from original data
    '''

    if not isinstance(data, pl.DataFrame):
        raise TypeError('data must be a Polars DataFrame')
    
    if data.is_empty():
        raise ValueError('data cannot be empty')
    
    if 'datetime' not in data.columns:
        raise ValueError('data must contain datetime column')
    
    if sample_size <= 0:
        raise ValueError('sample_size must be positive')
    
    if n_samples <= 0:
        raise ValueError('n_samples must be positive')
    
    if not (0.0 <= safe_range_low < safe_range_high <= 1.0):
        raise ValueError('safe_range_low must be >= 0.0, safe_range_high must be <= 1.0, and safe_range_low < safe_range_high')
    
    datasets = []
    
    for i in range(n_samples):
        if seed is not None:
            rng = np.random.default_rng(seed=seed + i)
        else:
            rng = np.random.default_rng()
        
        n = len(data)
        lo = int(n * safe_range_low)
        hi = int(n * safe_range_high) - sample_size
        
        if hi < lo:
            raise ValueError(f'sample_size ({sample_size}) too large for chosen safe range ({safe_range_low*100:.0f}%-{safe_range_high*100:.0f}%)')
        
        start = int(rng.integers(lo, hi + 1))
        datasets.append(data[start:start + sample_size])
            
    return datasets