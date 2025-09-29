import polars as pl
import numpy as np
from typing import List, Dict


def quantile_line_density(data: pl.DataFrame,
                          long_lines_q: List[Dict],
                          short_lines_q: List[Dict],
                          lookback_hours: int) -> pl.DataFrame:
    
    '''
    Compute count of quantile-filtered line ends within lookback_hours window.

    Args:
        data (pl.DataFrame): Klines dataset with 'datetime' and 'close' columns
        long_lines_q (list[dict]): Quantile-filtered long lines with 'end_idx'
        short_lines_q (list[dict]): Quantile-filtered short lines with 'end_idx'
        lookback_hours (int): Window size in hours for density calculation
        
    Returns:
        pl.DataFrame: The input data with a new column 'quantile_line_density_48h'
    '''

    n_rows = data.height
    if n_rows == 0:
        return data.with_columns([pl.lit(0).alias('quantile_line_density_48h')])

    ends = np.array(sorted([line['end_idx'] for line in long_lines_q] + [s['end_idx'] for s in short_lines_q]))
    density = np.zeros(n_rows, dtype=int)
    if ends.size == 0:
        return data.with_columns([pl.Series('quantile_line_density_48h', density)])

    left = 0
    for i in range(n_rows):
        while left < len(ends) and ends[left] < i - lookback_hours:
            left += 1
        right = left
        while right < len(ends) and ends[right] <= i:
            right += 1
        density[i] = max(0, right - left)
    return data.with_columns([pl.Series('quantile_line_density_48h', density)])


