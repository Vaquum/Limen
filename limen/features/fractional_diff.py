import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def _get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:

    '''
    Compute Fixed-Width Fractional Differentiation (FFD) weight vector.

    Based on Lopez de Prado's AFML Chapter 5.
    Weights: w_0 = 1, w_k = -w_{k-1} * (d - k + 1) / k,
    truncated when |w_k| < threshold.

    Args:
        d (float): Fractional differentiation order
        threshold (float): Weight truncation threshold

    Returns:
        np.ndarray: Weight vector [w_0, w_1, ..., w_n]
    '''

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
        raise ValueError('cols must be a non-empty list of column names')

    if d < 0:
        raise ValueError(f"d must be non-negative, got {d}")

    is_lazy = isinstance(data, pl.LazyFrame)
    df = data.collect() if is_lazy else data

    if df.height == 0:
        return df.lazy() if is_lazy else df

    if d == 0.0:
        for col in cols:
            if col not in df.columns:
                logger.warning('Column %s not found, skipping fractional diff', col)
        new_cols = [
            df[col].cast(pl.Float64).alias(f"{col}_fracdiff")
            for col in cols if col in df.columns
        ]
        if new_cols:
            df = df.with_columns(new_cols)
        return df.lazy() if is_lazy else df

    weights = _get_weights_ffd(d, threshold)
    width = len(weights)

    result_cols: list[pl.Series] = []

    for col in cols:
        if col not in df.columns:
            logger.warning('Column %s not found, skipping fractional diff', col)
            continue

        try:
            values = df[col].cast(pl.Float64).to_numpy()
            n = len(values)
            output = np.full(n, np.nan)

            if width <= n:
                output[width - 1:] = np.convolve(values, weights[::-1], mode='valid')

            if np.all(np.isnan(output)):
                logger.warning(
                    'Fractional diff for column %s (d=%.2f) requires %d rows '
                    'but split has %d — column %s_fracdiff not added',
                    col, d, width, n, col,
                )
                continue

            series = pl.Series(f"{col}_fracdiff", output)
            result_cols.append(series.fill_nan(None))
        except (ValueError, TypeError, ArithmeticError):
            logger.warning(
                'Fractional diff failed for column %s (d=%.2f) — '
                'column %s_fracdiff not added',
                col, d, col,
                exc_info=True,
            )
            continue

    if result_cols:
        df = df.with_columns(result_cols)

    return df.lazy() if is_lazy else df


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
        raise ValueError(f"step must be positive, got {step}")

    fracdiff_col = f"{col}_fracdiff"

    d = d_start
    while d <= d_end:
        diffed = fractional_diff(data, d=d, cols=[col], threshold=threshold)

        if fracdiff_col not in diffed.columns:
            d = round(d + step, 10)
            continue

        series = diffed[fracdiff_col].drop_nulls().drop_nans()
        if len(series) > 0:
            result = adf_test(series, significance_level=significance_level)
            if result.stationary:
                return round(d, 10)

        d = round(d + step, 10)

    return d_end
