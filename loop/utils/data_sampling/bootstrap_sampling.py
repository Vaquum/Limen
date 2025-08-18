import polars as pl
import numpy as np
from typing import List, Optional


DEFAULT_SAMPLE_SIZE = 10000
DEFAULT_N_SAMPLES = 3


def bootstrap_sampling(data: pl.DataFrame,
                       sample_size: int = DEFAULT_SAMPLE_SIZE,
                       n_samples: int = DEFAULT_N_SAMPLES,
                       seed: Optional[int] = None) -> List[pl.DataFrame]:

    '''
    Compute bootstrap samples with replacement from dataset.

    Args:
        data (pl.DataFrame): Klines dataset with 'datetime' and numeric columns
        sample_size (int): Number of rows in each bootstrap sample
        n_samples (int): Number of bootstrap samples to create
        seed (int | None): Random seed for reproducible results
        
    Returns:
        List[pl.DataFrame]: List of bootstrap sampled datasets with replacement
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
    
    datasets = []
    total_rows = len(data)
    
    if sample_size > total_rows:
        raise ValueError(f'sample_size ({sample_size}) cannot exceed data length ({total_rows})')
    
    for i in range(n_samples):
        if seed is not None:
            rng = np.random.default_rng(seed=seed + i)
        else:
            rng = np.random.default_rng()
        
        indices = rng.choice(total_rows, size=sample_size, replace=True)
        sorted_indices = sorted(indices)
        datasets.append(data[sorted_indices])
        
    return datasets