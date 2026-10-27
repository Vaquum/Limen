from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import polars as pl

logger = logging.getLogger(__name__)
_MIN_CORRELATION_ROWS = 2

_FRAMEWORK_LOG_COLUMNS = frozenset({
    'id', '_id', '_round_index', 'execution_time', '_warnings',
})


def experiment_parameter_correlation(self: Any,
                                   metric: str,
                                   *,
                                   cols_to_drop: list[str] | None = None,
                                   sort_key: str | None = None,
                                   sort_ascending: bool = False,
                                   heads: Sequence[float | int] | None = None,
                                   method: Literal['pearson', 'kendall', 'spearman'] = 'spearman',
                                   n_boot: int = 300,
                                   min_n: int = 10,
                                   random_state: int = 0) -> pl.DataFrame:
    '''Return bootstrap correlations with explicit cohort_pct and feature columns.'''
    if method not in ('pearson', 'kendall', 'spearman'):
        raise ValueError('Correlation method must be pearson, kendall or spearman')
    source = self.experiment_log.drop(cols_to_drop or [])
    data_numeric = source.select(
        pl.col(name).cast(pl.Float64, strict=False).fill_nan(None)
        if dtype.is_numeric() or dtype in (pl.Boolean, pl.String, pl.Null)
        else pl.lit(None, dtype=pl.Float64).alias(name)
        for name, dtype in source.schema.items()
    )
    if heads is None:
        heads = (0.99, 0.75, 0.5, 0.25, 0.01)
    if sort_key is None:
        sort_key = metric
    if metric not in data_numeric.columns:
        raise ValueError(f'experiment_parameter_correlation metric "{metric}" not found in data columns')
    if sort_key not in data_numeric.columns:
        raise ValueError(f'experiment_parameter_correlation sort_key "{sort_key}" not found in data columns')
    df = data_numeric.drop_nulls(metric).sort(sort_key, descending=not sort_ascending,
                                              nulls_last=True, maintain_order=True)
    if df.is_empty():
        raise ValueError('experiment_parameter_correlation No rows remain after dropping NaNs in the metric column.')
    num_df = df.select(name for name in df.columns if df[name].drop_nulls().n_unique() > 1)
    if metric not in num_df.columns:
        raise ValueError('experiment_parameter_correlation After cleaning, metric is not numeric. Ensure it is numeric in self.experiment_log.')
    features = [name for name in num_df.columns if name != metric and name not in _FRAMEWORK_LOG_COLUMNS]
    if not features:
        raise ValueError('experiment_parameter_correlation No numeric features available (besides metric) after cleaning.')

    def correlations(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        from scipy.stats import kendalltau, rankdata
        target = matrix[:, 0]
        values: list[float] = []
        for column in matrix[:, 1:].T:
            present = np.isfinite(column) & np.isfinite(target)
            x, y = column[present], target[present]
            if x.size < _MIN_CORRELATION_ROWS or np.ptp(x) == 0 or np.ptp(y) == 0:
                values.append(np.nan)
            elif method == 'kendall':
                values.append(float(kendalltau(x, y).statistic))
            else:
                if method == 'spearman':
                    x, y = np.asarray(rankdata(x), dtype=np.float64), np.asarray(rankdata(y), dtype=np.float64)
                values.append(float(np.corrcoef(x, y)[0, 1]))
        return np.asarray(values, dtype=np.float64)

    rng = np.random.default_rng(random_state)
    total_n = num_df.height
    records: list[dict[str, object]] = []
    cohort_order: dict[float, int] = {}
    for head in heads:
        n = max(int(total_n * head), 1) if isinstance(head, float) and head <= 1.0 else int(head)
        n = min(n, total_n)
        if n < min_n:
            logger.debug('Skipping cohort (requested=%s) with n=%d < min_n=%d', head, n, min_n)
            continue
        percentage = 100 * n / total_n
        order = cohort_order.setdefault(percentage, len(cohort_order))
        cohort = np.asarray(num_df.head(n).select(metric, *features).to_numpy(), dtype=np.float64)
        base = correlations(cohort)
        boot = np.asarray([correlations(cohort[rng.integers(0, n, n)]) for _ in range(n_boot)], dtype=np.float64).T
        for index, name in enumerate(features):
            samples = boot[index]
            valid = samples[~np.isnan(samples)]
            median = float(cast(np.float64, np.quantile(valid, 0.5))) if valid.size else np.nan
            low = float(cast(np.float64, np.quantile(valid, 0.025))) if valid.size else np.nan
            high = float(cast(np.float64, np.quantile(valid, 0.975))) if valid.size else np.nan
            records.append({'cohort_pct': percentage, 'feature': name, 'n_rows': n,
                            'corr': float(base[index]), 'corr_med': float(median),
                            'ci_lo': float(low), 'ci_hi': float(high),
                            'sign_stability': float(np.count_nonzero(np.sign(samples) == np.sign(median)) / samples.size),
                            '_order': order})
    if not records:
        raise ValueError('experiment_parameter_correlation No cohorts produced results (check heads/min_n and data quality).')
    return pl.DataFrame(records).sort(['_order', 'feature'], maintain_order=True).drop('_order')
