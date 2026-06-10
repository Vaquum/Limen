import numpy as np

MIN_DURATION_BARS = 1
MIN_MAX_DURATION_HOURS = 2
MIN_PRICES = 2


def find_price_lines(close: np.ndarray,
                     max_duration_hours: int,
                     min_height_pct: float) -> tuple[list[dict], list[dict]]:

    '''
    Compute linear price movements (lines) from a close-price series.

    A line is any pair of bars (start, end) with duration in
    [1, max_duration_hours) whose relative close-to-close change is at
    least min_height_pct in absolute terms. Lines with a positive change
    are long lines, negative are short lines.

    Args:
        close (np.ndarray): One-dimensional array of close prices
        max_duration_hours (int): Exclusive upper bound on line duration in bars
        min_height_pct (float): Minimum absolute height as a fraction of the start price

    Returns:
        tuple[list[dict], list[dict]]: (long_lines, short_lines), each a list of
            dicts with keys 'start_idx', 'end_idx', 'height_pct', 'duration_hours'

    Raises:
        ValueError: If max_duration_hours is below 2 or min_height_pct is not positive

    NOTE: Lines are detected only within the array passed in, so per-split
    invocation cannot observe other splits.
    '''

    if max_duration_hours < MIN_MAX_DURATION_HOURS:
        raise ValueError('find_price_lines max_duration_hours must be at least 2')

    if min_height_pct <= 0:
        raise ValueError('find_price_lines min_height_pct must be positive')

    close = np.asarray(close, dtype=np.float64)
    n_prices = close.shape[0]

    long_lines: list[dict] = []
    short_lines: list[dict] = []

    if n_prices < MIN_PRICES:
        return long_lines, short_lines

    starts_per_duration = []
    heights_per_duration = []
    durations_per_duration = []

    for duration in range(MIN_DURATION_BARS, max_duration_hours):

        if duration >= n_prices:
            break

        heights = (close[duration:] - close[:-duration]) / close[:-duration]
        qualifying = np.nonzero(np.abs(heights) >= min_height_pct)[0]

        if qualifying.size == 0:
            continue

        starts_per_duration.append(qualifying)
        heights_per_duration.append(heights[qualifying])
        durations_per_duration.append(np.full(qualifying.size, duration))

    if not starts_per_duration:
        return long_lines, short_lines

    starts = np.concatenate(starts_per_duration)
    heights = np.concatenate(heights_per_duration)
    durations = np.concatenate(durations_per_duration)

    for start, height, duration in zip(starts, heights, durations, strict=True):
        line = {
            'start_idx': int(start),
            'end_idx': int(start + duration),
            'height_pct': float(height),
            'duration_hours': int(duration),
        }

        if height > 0:
            long_lines.append(line)
        else:
            short_lines.append(line)

    return long_lines, short_lines
