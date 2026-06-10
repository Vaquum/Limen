import numpy as np


def filter_lines_by_quantile(lines: list[dict], quantile: float) -> list[dict]:

    '''
    Compute the subset of lines at or above a height quantile.

    Args:
        lines (list[dict]): Line dicts with a 'height_pct' key, as produced
            by find_price_lines
        quantile (float): Quantile of absolute heights below which lines are dropped

    Returns:
        list[dict]: Lines whose absolute height is at or above the quantile cutoff

    Raises:
        ValueError: If quantile is outside [0, 1]
    '''

    if not 0 <= quantile <= 1:
        raise ValueError('filter_lines_by_quantile quantile must be between 0 and 1')

    if not lines:
        return []

    heights = np.abs(np.array([line['height_pct'] for line in lines]))
    cutoff = np.quantile(heights, quantile)

    return [line for line, height in zip(lines, heights, strict=True) if height >= cutoff]
