import polars as pl
import numpy as np
from typing import List, Dict


def active_quantile_count(data: pl.DataFrame,
                          long_lines_q: List[Dict],
                          short_lines_q: List[Dict]) -> pl.DataFrame:
    
    '''
    Compute count of active quantile-filtered lines per row.

    Args:
        data (pl.DataFrame): Klines dataset with 'datetime' and 'close' columns
        long_lines_q (list[dict]): Quantile-filtered long lines with 'start_idx' and 'end_idx'
        short_lines_q (list[dict]): Quantile-filtered short lines with 'start_idx' and 'end_idx'
        
    Returns:
        pl.DataFrame: The input data with a new column 'active_quantile_count'
    '''

    n_rows = data.height
    if n_rows == 0:
        return data.with_columns([pl.lit(0).alias('active_quantile_count')])

    events = []
    for line in long_lines_q:
        events.append((line['start_idx'], 1))
        events.append((line['end_idx'], -1))
    for line in short_lines_q:
        events.append((line['start_idx'], 1))
        events.append((line['end_idx'], -1))
    if not events:
        return data.with_columns([pl.lit(0).alias('active_quantile_count')])

    events.sort()
    active = np.zeros(n_rows, dtype=int)
    current = 0
    eidx = 0
    for idx in range(n_rows):
        while eidx < len(events) and events[eidx][0] <= idx:
            current += events[eidx][1]
            eidx += 1
        active[idx] = max(current, 0)
    return data.with_columns([pl.Series('active_quantile_count', active)])


