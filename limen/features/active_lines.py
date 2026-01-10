import polars as pl
import numpy as np
from typing import List, Dict


def active_lines(data: pl.DataFrame,
                 long_lines: List[Dict],
                 short_lines: List[Dict]) -> pl.DataFrame:
    
    '''
    Compute active line count per row from provided long and short line spans.

    Args:
        data (pl.DataFrame): Klines dataset with 'datetime' and 'close' columns
        long_lines (list[dict]): Long line definitions with 'start_idx' and 'end_idx'
        short_lines (list[dict]): Short line definitions with 'start_idx' and 'end_idx'
        
    Returns:
        pl.DataFrame: The input data with a new column 'active_lines'
    '''

    n_rows = data.height
    if n_rows == 0:
        return data.with_columns([pl.lit(0).alias('active_lines')])

    events = []
    for line in long_lines:
        events.append((line['start_idx'], 1))
        events.append((line['end_idx'] + 1, -1))
    for line in short_lines:
        events.append((line['start_idx'], 1))
        events.append((line['end_idx'] + 1, -1))
    if not events:
        return data.with_columns([pl.lit(0).alias('active_lines')])

    events.sort()
    active = np.zeros(n_rows, dtype=int)
    current = 0
    eidx = 0
    for idx in range(n_rows):
        while eidx < len(events) and events[eidx][0] <= idx:
            current += events[eidx][1]
            eidx += 1
        active[idx] = current

    return data.with_columns([pl.Series('active_lines', active)])


