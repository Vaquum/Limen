from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _experiment_parameter_correlation(self,
                                   metric: str,
                                   *,
                                   cols_to_drop: Optional[List] = None,
                                   sort_key: Optional[str] = None,
                                   sort_ascending=False,
                                   heads: Optional[Sequence[Union[float, int]]] = None,
                                   method: str = 'spearman',
                                   n_boot: int = 300,
                                   min_n: int = 10,
                                   random_state: int = 0) -> pd.DataFrame:
    
    '''
    Compute robust correlations between parameters and metrics across explicit cohorts.

    Args:
        metric (str): Target column to correlate against (e.g., 'auc')
        sort_key (str | None): Column used to rank rows before slicing
        sort_ascending (bool): Whether to sort ascending when creating cohorts
        heads (Sequence[float | int] | None): Cohort sizes; fractions (0, 1] as proportions or integers as counts
        method (str): Correlation method: 'spearman', 'pearson', or 'kendall'
        n_boot (int): Number of bootstrap resamples per cohort
        min_n (int): Minimum cohort size; smaller cohorts are skipped
        random_state (int): RNG seed for reproducibility

    Returns:
        pd.DataFrame: MultiIndex rows indexed by ('cohort_pct', 'feature') with columns
                      'n_rows', 'corr', 'corr_med', 'ci_lo', 'ci_hi', 'sign_stability'
    
    NOTE: Non-numeric columns are coerced with errors='coerce' and ignored thereafter;
          constant or all-NaN numeric columns are dropped; rows with NaN in `metric` are
          dropped prior to sorting and slicing
    '''

    if cols_to_drop is not None:
        data_numeric = self.experiment_log.copy()
        data_numeric.drop(cols_to_drop, axis=1, inplace=True)
    
    else:
        data_numeric = self.experiment_log.copy()
    
    if heads is None:
        heads = (0.99, 0.75, 0.5, 0.25, 0.01)
    
    if sort_key is None:
        sort_key = metric
    
    for c in data_numeric.columns:
        data_numeric[c] = pd.to_numeric(data_numeric[c], errors='coerce')

    if metric not in data_numeric.columns:
        raise ValueError(f'metric "{metric}" not found in data columns')
    
    if sort_key not in data_numeric.columns:
        raise ValueError(f'sort_key "{sort_key}" not found in data columns')

    df = (
        data_numeric
        .dropna(subset=[metric])
        .sort_values(sort_key, ascending=sort_ascending)
        .reset_index(drop=True)
    )
    
    if df.empty:
        raise ValueError('No rows remain after dropping NaNs in the metric column.')

    num_df = df.select_dtypes(include=[np.number])
    nunique = num_df.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    
    if constant_cols:
        num_df = num_df.drop(columns=constant_cols, errors='ignore')

    if metric not in num_df.columns:
        raise ValueError('After cleaning, metric is not numeric. Ensure it is numeric in self.experiment_log.')

    features: List[str] = [c for c in num_df.columns if c != metric]
    if not features:
        raise ValueError('No numeric features available (besides metric) after cleaning.')

    rng = np.random.default_rng(random_state)
    total_n = len(num_df)

    def _corr_series(d: pd.DataFrame) -> pd.Series:
        corr_mat = d[features + [metric]].corr(method=method)
        s = corr_mat.get(metric)
        if s is None:
            return pd.Series(dtype=float)
        return s.drop(labels=[metric], errors='ignore')

    blocks: List[pd.DataFrame] = []
    realized_pcts: List[int] = []

    for h in heads:
        if isinstance(h, float) and h <= 1.0:
            n = max(int(total_n * h), 1)
        else:
            n = int(h)
        n = min(n, total_n)

        if n < min_n:
            logger.debug(f'Skipping cohort (requested={h}) with n={n} < min_n={min_n}')
            continue

        cohort = num_df.iloc[:n]
        base = _corr_series(cohort)
        if base.empty:
            logger.debug(f'No correlations computed for cohort (requested={h}). Skipping.')
            continue

        boot_vals: List[pd.Series] = []
        for _ in range(n_boot):
            sample_idx = rng.integers(0, n, n)
            s = _corr_series(cohort.iloc[sample_idx]).reindex(base.index)
            boot_vals.append(s)

        boot_df = pd.concat(boot_vals, axis=1)
        med = boot_df.median(axis=1)
        lo = boot_df.quantile(0.025, axis=1)
        hi = boot_df.quantile(0.975, axis=1)
        median_sign = np.sign(med)
        sign_stability = np.sign(boot_df).eq(median_sign, axis=0).mean(axis=1)

        cohort_pct = int(round(100 * n / total_n))
        realized_pcts.append(cohort_pct)

        block = pd.DataFrame(
            {
                'feature': base.index,
                'cohort_pct': cohort_pct,   # integer 0..100
                'n_rows': n,
                'corr': base.values,
                'corr_med': med.values,
                'ci_lo': lo.values,
                'ci_hi': hi.values,
                'sign_stability': sign_stability.values,
            }
        )
        blocks.append(block)

    if not blocks:
        raise ValueError('No cohorts produced results (check heads/min_n and data quality).')

    res = pd.concat(blocks, ignore_index=True)

    order = []
    seen = set()
    for pct in realized_pcts:
        if pct not in seen:
            order.append(pct)
            seen.add(pct)

    res['cohort_pct'] = pd.Categorical(res['cohort_pct'], categories=order, ordered=True)
    res = (
        res.assign(abs_corr_med=res['corr_med'].abs())
           .sort_values(['cohort_pct', 'abs_corr_med'], ascending=[True, False])
           .drop(columns='abs_corr_med')
           .set_index(['cohort_pct', 'feature'])
           .sort_index(level=0, sort_remaining=True)
    )

    return res
