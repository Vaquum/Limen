import polars as pl
import numpy as np
from typing import List
from typing import Dict


def hours_since_quantile_line(data: pl.DataFrame,
                              long_lines_q: List[Dict],
                              short_lines_q: List[Dict],
                              lookback_hours: int) -> pl.DataFrame:
    '''
    Compute hours since the most recent quantile-filtered line end capped by lookback_hours.

    Args:
        data (pl.DataFrame): Klines dataset with 'datetime' and 'close' columns
        long_lines_q (list[dict]): Quantile-filtered long lines with 'end_idx'
        short_lines_q (list[dict]): Quantile-filtered short lines with 'end_idx'
        lookback_hours (int): Maximum number of hours to track recency

    Returns:
        pl.DataFrame: The input data with a new column 'hours_since_quantile_line'
    '''

    n_rows = data.height

    if n_rows == 0:
        return data.with_columns([pl.lit(float(lookback_hours)).alias('hours_since_quantile_line')])

    ends = [line['end_idx'] for line in long_lines_q] + [s['end_idx'] for s in short_lines_q]

    if not ends:
        return data.with_columns([pl.lit(float(lookback_hours)).alias('hours_since_quantile_line')])

    ends = np.array(sorted(ends))
    hours_since = np.full(n_rows, float(lookback_hours))
    ptr = -1

    for i in range(n_rows):
        while ptr + 1 < len(ends) and ends[ptr + 1] <= i:
            ptr += 1

        if ptr >= 0:
            hours_since[i] = float(min(i - ends[ptr], lookback_hours))

    return data.with_columns([pl.Series('hours_since_quantile_line', hours_since)])
