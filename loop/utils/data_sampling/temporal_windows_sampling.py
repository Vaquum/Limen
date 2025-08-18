import polars as pl
from typing import List


DEFAULT_WINDOW_SIZE = 10000
DEFAULT_OVERLAP = 0.2
MAX_WINDOWS = 10


def temporal_windows_sampling(data: pl.DataFrame,
                              window_size: int = DEFAULT_WINDOW_SIZE,
                              overlap: float = DEFAULT_OVERLAP) -> List[pl.DataFrame]:

    '''
    Compute overlapping temporal windows from chronologically ordered dataset.

    Args:
        data (pl.DataFrame): Klines dataset with 'datetime' and numeric columns
        window_size (int): Size of each temporal window in rows
        overlap (float): Fraction of overlap between consecutive windows
        
    Returns:
        List[pl.DataFrame]: List of overlapping temporal window datasets
    '''

    if not isinstance(data, pl.DataFrame):
        raise TypeError('data must be a Polars DataFrame')
    
    if data.is_empty():
        raise ValueError('data cannot be empty')
    
    if 'datetime' not in data.columns:
        raise ValueError('data must contain datetime column')
    
    if window_size <= 0:
        raise ValueError('window_size must be positive')
    
    if not (0.0 <= overlap < 1.0):
        raise ValueError('overlap must be >= 0.0 and < 1.0')
    
    total_rows = len(data)
    
    if total_rows < window_size:
        return []
    
    datasets = []
    step_size = int(window_size * (1 - overlap))
    
    start_positions = []
    i = 0
    while True:
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        if end_idx > total_rows:
            break
        else:
            start_positions.append(start_idx)
            i += 1
            
        if len(start_positions) >= MAX_WINDOWS:
            break
    
    for start_idx in start_positions:
        end_idx = start_idx + window_size
        datasets.append(data[start_idx:end_idx])
        
    return datasets