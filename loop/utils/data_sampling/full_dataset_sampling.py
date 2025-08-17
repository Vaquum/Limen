import polars as pl
from typing import List


def full_dataset_sampling(data: pl.DataFrame) -> List[pl.DataFrame]:

    '''
    Compute full dataset as baseline comparison for data sampling strategies.

    Args:
        data (pl.DataFrame): Klines dataset with 'datetime' and numeric columns
        
    Returns:
        List[pl.DataFrame]: Single-item list containing the complete original dataset
    '''

    if not isinstance(data, pl.DataFrame):
        raise TypeError('data must be a Polars DataFrame')
    
    if data.is_empty():
        raise ValueError('data cannot be empty')
    
    if 'datetime' not in data.columns:
        raise ValueError('data must contain datetime column')
    
    return [data]