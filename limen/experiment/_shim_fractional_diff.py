from __future__ import annotations

import polars as pl

from limen.backtest.trade_contract import finite_number
from limen.features.fractional_diff import FRACDIFF_SUFFIX, find_min_d, fractional_diff
from limen.utils.adf_test import adf_test

__all__ = ['fit_order', 'fractional_column']


def fractional_column(data: pl.DataFrame, column: str, order: float, threshold: float) -> pl.Series:
    return fractional_diff(data, d=order, cols=[column], threshold=threshold)[f'{column}{FRACDIFF_SUFFIX}']


def fit_order(
    data: pl.DataFrame, column: str, declared: object, threshold: float, significance_level: float,
) -> float:
    if declared != 'auto':
        return finite_number(declared, 'feature_shims d')
    order = find_min_d(
        data, col=column, d_start=0.0, d_end=1.0, step=0.1,
        threshold=threshold, significance_level=significance_level,
    )
    candidate = fractional_column(data, column, order, threshold)
    if not adf_test(candidate, significance_level=significance_level).stationary:
        raise ValueError(f'feature_shims no qualifying fractional-difference order for {column!r}')
    return order
