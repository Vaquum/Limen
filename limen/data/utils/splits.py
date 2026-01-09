import polars as pl

from typing import Sequence, List
from itertools import accumulate


def split_sequential(data: pl.DataFrame, ratios: Sequence[int]) -> List[pl.DataFrame]:
    
    '''
    Compute sequential data splits with proportional lengths based on ratios.
    
    Args:
        data (pl.DataFrame): Polars DataFrame to split sequentially
        ratios (Sequence[int]): Sequence of positive integers defining split proportions
        
    Returns:
        List[pl.DataFrame]: List of DataFrames partitioned sequentially without losing or duplicating rows
    '''
    
    total = data.height
    if total == 0:
        return [pl.DataFrame() for _ in ratios]

    total_ratio = sum(ratios)
    
    sizes: List[int] = []
    cumulative = 0
    
    for r in ratios[:-1]:
        chunk_size = int(total * r / total_ratio)
        sizes.append(chunk_size)
        cumulative += chunk_size

    sizes.append(total - cumulative)

    out: List[pl.DataFrame] = []
    start = 0
    for size in sizes:
        out.append(data.slice(start, size))
        start += size

    return out


def split_random(data: pl.DataFrame, ratios: Sequence[int], seed: int = None) -> List[pl.DataFrame]:
    
    '''
    Compute random data splits with proportional lengths based on ratios.
    
    Args:
        data (pl.DataFrame): Polars DataFrame to split randomly
        ratios (Sequence[int]): Sequence of positive integers defining split proportions
        seed (int): Seed for random number generator
        
    Returns:
        List[pl.DataFrame]: List of randomly shuffled DataFrames with proportional sizes
    '''

    total = data.height
    total_ratio = sum(ratios)
    bounds = [int(total * c / total_ratio) for c in accumulate(ratios)]
    starts = [0] + bounds[:-1]
    
    return [data.sample(fraction=1.0, seed=seed, shuffle=True).slice(start, end - start) for start, end in zip(starts, bounds)]


def split_data_to_prep_output(split_data: list,
                              cols: list,
                              all_datetimes: list) -> dict:
    
    '''
    Compute data preparation output dictionary from split data and column names.
    
    Args:
        split_data (list): List of three DataFrames representing train, validation, and test splits
        cols (list): Column names where the last column is the target variable
        all_datetimes (list): List of all datetimes
        
    Returns:
        dict: Dictionary with train, validation, and test features and targets
    '''

    remaining_datetimes = split_data[0]['datetime'].to_list()
    remaining_datetimes += split_data[1]['datetime'].to_list()
    remaining_datetimes += split_data[2]['datetime'].to_list()

    first_test_datetime = split_data[2]['datetime'].min()
    last_test_datetime = split_data[2]['datetime'].max()

    split_data[0] = split_data[0].drop('datetime')
    split_data[1] = split_data[1].drop('datetime')
    split_data[2] = split_data[2].drop('datetime')

    if 'datetime' in cols:
        cols.remove('datetime')
    else:
        raise ValueError('SFDs must contain `datetime` in data up to when it enters `split_data_to_prep_output` in sfd.prep')
            
    data_dict = {'x_train': split_data[0][cols[:-1]],
                 'y_train': split_data[0][cols[-1]],
                 'x_val': split_data[1][cols[:-1]],
                 'y_val': split_data[1][cols[-1]],
                 'x_test': split_data[2][cols[:-1]],
                 'y_test': split_data[2][cols[-1]]}

    data_dict['_alignment'] = {}
    
    data_dict['_alignment']['missing_datetimes'] = sorted(set(all_datetimes) - set(remaining_datetimes))
    data_dict['_alignment']['first_test_datetime'] = first_test_datetime
    data_dict['_alignment']['last_test_datetime'] = last_test_datetime
    
    return data_dict
