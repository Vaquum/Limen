from typing import Sequence, List
from itertools import accumulate

import polars as pl


def split_sequential(data, ratios: Sequence[int]) -> List[pl.DataFrame]:

    '''Split the data into sequential chunks

    Args:
        ratios (Sequence[int]): The ratios of the data to be split

    Returns:
        List[pl.DataFrame]
    '''

    total = data.height
    total_ratio = sum(ratios)
    bounds = [int(total * c / total_ratio) for c in accumulate(ratios)]
    starts = [0] + bounds[:-1]
    
    return [data.slice(start, end - start) for start, end in zip(starts, bounds)]


def split_random(data, ratios: Sequence[int], seed: int = None) -> List[pl.DataFrame]:

    '''Split the data into random chunks

    Args:
        ratios (Sequence[int]): The ratios of the data to be split
        seed (int): The seed for the random number generator

    Returns:
        List[pl.DataFrame]    
    '''

    total = data.height
    total_ratio = sum(ratios)
    bounds = [int(total * c / total_ratio) for c in accumulate(ratios)]
    starts = [0] + bounds[:-1]
    
    return [data.sample(fraction=1.0, seed=seed, shuffle=True).slice(start, end - start) for start, end in zip(starts, bounds)]


def split_data_to_prep_output(split_data, cols):

    return {
        'x_train': split_data[0][cols[:-1]],
        'y_train': split_data[0][cols[-1]],
        'x_val': split_data[1][cols[:-1]],
        'y_val': split_data[1][cols[-1]],
        'x_test': split_data[2][cols[:-1]],
        'y_test': split_data[2][cols[-1]],
    }