from typing import Sequence, List
from itertools import accumulate

import polars as pl


from typing import Sequence, List
from itertools import accumulate

import polars as pl


def split_sequential(data: pl.DataFrame, ratios: Sequence[int]) -> List[pl.DataFrame]:
    """
    Split `data` into sequential chunks whose lengths are proportional to `ratios`,
    without ever losing or duplicating rows.

    Example:
        If data.height = 11 and ratios = [2, 3, 5], then
          total_ratio = 10,
          sizes = [ int(11*2/10)=2, int(11*3/10)=3, last = 11-(2+3)=6 ],
        so you return slices of lengths [2, 3, 6] in order.

    Args:
        data: a Polars DataFrame of length N = data.height
        ratios: a sequence of positive integers (or floats) whose sum is total_ratio

    Returns:
        A list of len(ratios) DataFrames, partitioned sequentially.
    """
    total = data.height
    if total == 0:
        return [pl.DataFrame() for _ in ratios]

    total_ratio = sum(ratios)
    # Compute the size of each chunk, except the last one “takes whatever is left.”
    sizes: List[int] = []
    cumulative = 0
    # For each ratio except the last, do a floor; the final one is total - sum(others).
    for r in ratios[:-1]:
        chunk_size = int(total * r / total_ratio)
        sizes.append(chunk_size)
        cumulative += chunk_size

    # Last chunk absorbs any leftover
    sizes.append(total - cumulative)

    # Now build the actual slices:
    out: List[pl.DataFrame] = []
    start = 0
    for size in sizes:
        # If size is zero, slice(…, 0) returns an empty DataFrame.
        out.append(data.slice(start, size))
        start += size

    return out


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