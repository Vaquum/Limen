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
        data (pl.DataFrame | pl.LazyFrame): Klines dataset with a close price column
        threshold (float): Cumulative log-return magnitude that triggers an event
        close_col (str): Column name used for close-to-close log returns
        output_col (str): Output column name

    Returns:
        pl.DataFrame | pl.LazyFrame: The input data, matching the input frame
            type, with an Int8 column: 1 for an up event, -1 for a down event,
            0 otherwise

    NOTE: Column access is expression-based so the feature runs on both eager
    DataFrames and the LazyFrames piped by the manifest feature pipeline.
    '''

    if threshold <= 0:
        raise ValueError('cusum_filter threshold must be positive')

    return data.with_columns(
        pl.col(close_col)
        .cast(pl.Float64)
        .map_batches(
            lambda s: pl.Series(_cusum_events(s.to_numpy(), threshold), dtype=pl.Int8),
            return_dtype=pl.Int8,
        )
        .alias(output_col)
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
