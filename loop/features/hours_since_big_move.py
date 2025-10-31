import polars as pl
import numpy as np
from typing import List
from typing import Dict


def hours_since_big_move(data: pl.DataFrame,
                         long_lines: List[Dict],
                         short_lines: List[Dict],
                         lookback_hours: int) -> pl.DataFrame:
    '''
    Compute hours since the most recent line end capped by lookback_hours.

    Args:
        data (pl.DataFrame): Klines dataset with 'datetime' and 'close' columns
        long_lines (list[dict]): Long line definitions with 'start_idx' and 'end_idx'
        short_lines (list[dict]): Short line definitions with 'start_idx' and 'end_idx'
        lookback_hours (int): Maximum number of hours to track recency

    Returns:
        pl.DataFrame: The input data with a new column 'hours_since_big_move'
    '''

    n_rows = data.height

    if n_rows == 0:
        return data.with_columns([pl.lit(float(lookback_hours)).alias('hours_since_big_move')])

    ended = [line['end_idx'] for line in long_lines] + [s['end_idx'] for s in short_lines]

    if not ended:
        return data.with_columns([pl.lit(float(lookback_hours)).alias('hours_since_big_move')])

    ended = np.array(sorted(ended))
    recency = np.full(n_rows, float(lookback_hours))
    ptr = -1

    for idx in range(n_rows):
        while ptr + 1 < len(ended) and ended[ptr + 1] < idx:
            ptr += 1

        if ptr >= 0:
            recency[idx] = float(min(idx - ended[ptr], lookback_hours))

    return data.with_columns([pl.Series('hours_since_big_move', recency)])
