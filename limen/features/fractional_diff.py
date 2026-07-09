import logging

import numpy as np
import numpy.typing as npt
import polars as pl

logger = logging.getLogger(__name__)

FRACDIFF_SUFFIX = '_fracdiff'


def _get_weights_ffd(d: float, threshold: float = 1e-5) -> npt.NDArray[np.float64]:

    '''
    Compute Fixed-Width Fractional Differentiation (FFD) weight vector.

    Based on Lopez de Prado's AFML Chapter 5.
    Weights: w_0 = 1, w_k = -w_{k-1} * (d - k + 1) / k,
    truncated when |w_k| < threshold.

    Args:
        d (float): Fractional differentiation order
        threshold (float): Weight truncation threshold

    Returns:
        np.ndarray: Weight vector [w_n, ..., w_1, w_0], where w_0 = 1.0 is the last element
    '''

    if threshold <= 0:
        raise ValueError(f"fractional_diff threshold must be positive, got {threshold}")

    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1

    return np.array(weights[::-1])


def fractional_diff(data: pl.DataFrame,
                    d: float = 0.0,
                    cols: list[str] | None = None,
                    threshold: float = 1e-5) -> pl.DataFrame:

    '''
    Apply Fixed-Width Fractional Differentiation (FFD) to specified columns.

    Based on Lopez de Prado's AFML Chapter 5. Achieves stationarity
    while preserving memory in financial time series. Adds new columns
    with suffix '_fracdiff' — original columns are preserved for
    indicators that depend on raw values.

    NOTE: AFML recommends applying to log-transformed prices for best results.

    Args:
        data (pl.DataFrame): Input data
        d (float): Fractional differentiation order (0 = identity copy, 1 = standard diff)
        cols (list[str] | None): Columns to differentiate (required)
        threshold (float): Weight truncation threshold per AFML standard

    Returns:
        pl.DataFrame: Data with new '{col}_fracdiff' columns added.
            When d=0, the new columns are copies of the originals.

    Raises:
        ValueError: If cols is None or empty
    '''

    if not cols:
        raise ValueError('fractional_diff cols must be a non-empty list of column names')

    if d < 0:
        raise ValueError(f"fractional_diff d must be non-negative, got {d}")

    if threshold <= 0:
        raise ValueError(f"fractional_diff threshold must be positive, got {threshold}")

    schema_names = data.collect_schema().names()

    if d == 0.0:
        for col in cols:
            if col not in schema_names:
                logger.warning('Column %s not found, skipping fractional diff', col)
        new_cols = [
            pl.col(col).cast(pl.Float64).alias(f"{col}{FRACDIFF_SUFFIX}")
            for col in cols if col in schema_names
        ]
        if new_cols:
            data = data.with_columns(new_cols)
        return data

    weights = _get_weights_ffd(d, threshold)

    new_cols = []
    for col in cols:
        if col not in schema_names:
            logger.warning('Column %s not found, skipping fractional diff', col)
            continue

        w = weights
        new_cols.append(
            pl.col(col).cast(pl.Float64).map_batches(
                lambda s, _w=w: _ffd_convolve(s, _w),
                return_dtype=pl.Float64,
            ).alias(f"{col}{FRACDIFF_SUFFIX}")
        )

    if new_cols:
        data = data.with_columns(new_cols)

    return data


def _ffd_convolve(series: pl.Series, weights: npt.NDArray[np.float64]) -> pl.Series:

    '''
    Apply FFD convolution to a single series.

    Args:
        series (pl.Series): Input values
        weights (np.ndarray): FFD weight vector from _get_weights_ffd

    Returns:
        pl.Series: Convolved values with leading nulls
    '''

    values = series.to_numpy()
    n = len(values)
    width = len(weights)
    output = np.full(n, np.nan)

    if width <= n:
        output[width - 1:] = np.convolve(values, weights[::-1], mode='valid')

    return pl.Series(output).fill_nan(None)


def find_min_d(data: pl.DataFrame,
               col: str,
               d_start: float = 0.0,
               d_end: float = 1.0,
               step: float = 0.1,
               significance_level: float = 0.05,
               threshold: float = 1e-5) -> float:

    '''
    Find the minimum fractional differentiation order achieving stationarity.

    Iterates from d_start to d_end by step, applying fractional
    differentiation at each d and testing with ADF. Returns the
    smallest d where the series is stationary.

    Args:
        data (pl.DataFrame): Input data containing the column to test
        col (str): Column name to differentiate and test
        d_start (float): Starting differentiation order
        d_end (float): Maximum differentiation order
        step (float): Increment between tested d values
        significance_level (float): ADF test significance threshold
        threshold (float): Weight truncation threshold for fractional diff

    Returns:
        float: Smallest d achieving stationarity, or d_end if none found
    '''

    from limen.utils.adf_test import adf_test

    if step <= 0:
        raise ValueError(f"fractional_diff step must be positive, got {step}")

    fracdiff_col = f"{col}{FRACDIFF_SUFFIX}"

    d = d_start
    while d <= d_end:
        diffed = fractional_diff(data, d=d, cols=[col], threshold=threshold)

        if fracdiff_col not in diffed.collect_schema().names():
            d = round(d + step, 10)
            continue

        series = diffed[fracdiff_col]
        if series.drop_nulls().len() > 0:
            result = adf_test(series, significance_level=significance_level)
            if result.stationary:
                return round(d, 10)

        d = round(d + step, 10)

    return d_end
