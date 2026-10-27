from decimal import Decimal

import numpy as np
import numpy.typing as npt
import polars as pl
import limen._feature_constraints as _check

FRACDIFF_SUFFIX = '_fracdiff'


def _get_weights_ffd(d: float, threshold: float = 1e-5) -> npt.NDArray[np.float64]:

    '''
    Compute Fixed-Width Fractional Differentiation (FFD) weight vector.

    Based on Lopez de Prado's AFML Chapter 5.
    Weights: w_0 = 1, w_k = -w_{k-1} * (d - k + 1) / k,
    truncated when |w_k| < threshold.

    Args:
        d (float): Finite, non-negative fractional differentiation order
        threshold (float): Finite, positive weight truncation threshold

    Returns:
        np.ndarray: Weight vector [w_n, ..., w_1, w_0], where w_0 = 1.0 is the last element

    Raises:
        ValueError: For invalid parameters or a non-finite recurrence weight.
    '''

    if not (isinstance(d, int) or np.isfinite(d)) or d < 0:
        raise ValueError(f"fractional_diff d must be finite and non-negative, got {d}")

    if not (isinstance(threshold, int) or np.isfinite(threshold)) or threshold <= 0:
        raise ValueError(f"fractional_diff threshold must be finite and positive, got {threshold}")

    weights = [1.0]
    k = 1
    with np.errstate(over='ignore'):
        while True:
            w = -weights[-1] * (d - k + 1) / k
            if not np.isfinite(w):
                raise ValueError(f"fractional_diff weight must be finite, got {w} at index {k}")
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
        d (float): Finite, non-negative order (0 = identity copy, 1 = standard diff)
        cols (list[str] | None): Columns to differentiate (required)
        threshold (float): Finite, positive weight truncation threshold per AFML standard

    Returns:
        pl.DataFrame: Data with new '{col}_fracdiff' columns added.
            When d=0, the new columns are copies of the originals.

    Raises:
        ValueError: If cols is None, empty, a string, or names missing columns,
            d is non-finite or negative, threshold is non-finite or nonpositive,
            or a recurrence weight is non-finite.
    '''

    if isinstance(cols, str) or not cols:
        raise ValueError('fractional_diff cols must be a non-empty list of column names')
    _check.numbers(context='fractional_diff', lower=0, d=d)
    _check.numbers(context='fractional_diff', positive=True, threshold=threshold)
    _check.column_sequence('cols', cols)


    if not (isinstance(d, int) or np.isfinite(d)) or d < 0:
        raise ValueError(f"fractional_diff d must be finite and non-negative, got {d}")

    if not (isinstance(threshold, int) or np.isfinite(threshold)) or threshold <= 0:
        raise ValueError(f"fractional_diff threshold must be finite and positive, got {threshold}")

    schema_names = data.collect_schema().names()
    missing = [col for col in cols if col not in schema_names]
    if missing:
        raise ValueError(f"fractional_diff columns not found in input schema: {missing}")

    if d == 0.0:
        new_cols = [
            pl.col(col).cast(pl.Float64).alias(f"{col}{FRACDIFF_SUFFIX}")
            for col in cols
        ]
        return data.with_columns(new_cols)

    weights = _get_weights_ffd(d, threshold)

    new_cols: list[pl.Expr] = []
    for col in cols:
        w = weights
        new_cols.append(
            pl.col(col).cast(pl.Float64).map_batches(
                lambda s, _w=w: _ffd_convolve(s, _w),
                return_dtype=pl.Float64,
            ).alias(f"{col}{FRACDIFF_SUFFIX}")
        )

    return data.with_columns(new_cols)


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

    _check.numbers(lower=0, d_start=d_start, d_end=d_end)
    _check.numbers(positive=True, step=step, threshold=threshold)
    _check.numbers(lower=0, upper=1, significance_level=significance_level)
    _check.names(col=col)
    _check.ordered('d_start', d_start, 'd_end', d_end)

    from limen.utils.adf_test import adf_test

    if step <= 0:
        raise ValueError(f"fractional_diff step must be positive, got {step}")

    fracdiff_col = f"{col}{FRACDIFF_SUFFIX}"

    start, end, increment = (Decimal(str(value)) for value in (d_start, d_end, step))
    count = int((end - start) // increment) + 1 if end >= start else 0
    for index in range(count):
        d = float(start + index * increment)
        diffed = fractional_diff(data, d=d, cols=[col], threshold=threshold)

        series = diffed[fracdiff_col]
        if series.drop_nulls().len() > 0:
            result = adf_test(series, significance_level=significance_level)
            if result.stationary:
                return d

    return d_end
