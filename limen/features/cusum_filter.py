import math

import numpy as np
import polars as pl


def cusum_filter(
    data: pl.DataFrame,
    threshold: float = 0.01,
    close_col: str = 'close',
    output_col: str = 'cusum_event',
) -> pl.DataFrame:

    '''
    Flag symmetric CUSUM events on the close-price log-return path.

    Running sums accumulate positive and negative log returns and reset when
    either breaches `threshold`, emitting an event at that bar (Lopez de Prado's
    symmetric CUSUM filter). The filter samples only moves whose cumulative
    magnitude is meaningful, gating out micro-noise ahead of downstream sampling.

    Args:
        data (pl.DataFrame): Klines dataset with a close price column
        threshold (float): Cumulative log-return magnitude that triggers an event
        close_col (str): Column name used for close-to-close log returns
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with an Int8 column: 1 for an up event,
            -1 for a down event, 0 otherwise
    '''

    if threshold <= 0:
        raise ValueError('cusum_filter threshold must be positive')

    close = data[close_col].to_numpy().astype(float, copy=False)
    return data.with_columns(
        pl.Series(output_col, _cusum_events(close, threshold), dtype=pl.Int8)
    )


def _cusum_events(close: np.ndarray, threshold: float) -> np.ndarray:
    n = close.shape[0]
    events = np.zeros(n, dtype=np.int8)
    s_pos = 0.0
    s_neg = 0.0

    for i in range(1, n):
        log_return = math.log(close[i]) - math.log(close[i - 1])
        s_pos = max(0.0, s_pos + log_return)
        s_neg = min(0.0, s_neg + log_return)
        if s_neg < -threshold:
            s_neg = 0.0
            events[i] = -1
        elif s_pos > threshold:
            s_pos = 0.0
            events[i] = 1

    return events
