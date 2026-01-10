import pandas as pd
import numpy as np
from typing import Sequence, Optional, Dict, Any
from math import sqrt


def _permutation_confusion_metrics(self,
                                  x: str,
                                  round_id: int,
                                  *,
                                  pred_col: str = 'predictions',
                                  actual_col: str = 'actuals',
                                  proba_col: Optional[str] = None,
                                  threshold: float = 0.5,
                                  outlier_quantiles: Sequence[float] = (0.01, 0.99),
                                  outlier_mode: str = 'filter',
                                  id_cols: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    '''
    Compute a single-row trimmed report for long-only evaluation: precision,
    recall, signal rates, counts and payoffs, and TP–FP separation.

    Args:
        x (str): Column summarized within TP/FP/TN/FN (e.g., predicted_probability or P&L)
        pred_col (str): Binary predictions column
        actual_col (str): Binary actuals column
        proba_col (str | None): Probabilities to binarize via `threshold` (overrides `pred_col`)
        threshold (float): Decision threshold for `proba_col`
        outlier_quantiles (Sequence[float]): (lo, hi) for x outlier handling
        outlier_mode (str): 'filter' to drop outside bounds or 'winsor' to clip
        id_cols (dict[str, Any] | None): Optional identifiers to prepend (e.g., params)

    Returns:
        pd.DataFrame: One-row table with columns 'x_name', 'n_kept', 'pred_pos_rate_pct',
                      'actual_pos_rate_pct', 'precision_pct', 'recall_pct', 'pred_pos_count',
                      'tp_count', 'fp_count', 'tp_x_mean', 'tp_x_median', 'fp_x_mean', 'fp_x_median',
                      'pred_pos_x_mean', 'pred_pos_x_median', 'tp_fp_cohen_d', 'tp_fp_ks'
    '''
    
    df = self.permutation_prediction_performance(round_id)

    # Optional: binarize from probabilities
    if proba_col is not None:
        if proba_col not in df:
            raise ValueError(f'proba_col "{proba_col}" not found')
    
        df[pred_col] = (df[proba_col].astype(float) >= float(threshold)).astype(int)

    # Validate required columns
    for col in (pred_col, actual_col, x):
        if col not in df:
            raise ValueError(f'column "{col}" not found')
    
    df[pred_col] = df[pred_col].astype(int)
    df[actual_col] = df[actual_col].astype(int)

    q_lo, q_hi = df[x].quantile(outlier_quantiles)
    if outlier_mode == 'filter':
        df = df[(df[x] >= q_lo) & (df[x] <= q_hi)]
    
    elif outlier_mode == 'winsor':
        df[x] = df[x].clip(q_lo, q_hi)
    
    else:
        raise ValueError('outlier_mode must be "filter" or "winsor"')

    n = len(df)
    if n == 0:
        raise ValueError('no rows remain after outlier handling')

    pred = df[pred_col]
    act = df[actual_col]
    m_tp = (pred == 1) & (act == 1)
    m_fp = (pred == 1) & (act == 0)
    m_tn = (pred == 0) & (act == 0)
    m_fn = (pred == 0) & (act == 1)

    def _stats(mask: pd.Series) -> Dict[str, float]:
        s = df.loc[mask, x]
        k = int(mask.sum())
        return {
            'count': k,
            'mean': float(s.mean()) if k else np.nan,
            'median': float(s.median()) if k else np.nan,
        }

    tp = _stats(m_tp)
    fp = _stats(m_fp)
    tn = _stats(m_tn)
    fn = _stats(m_fn)

    TP, FP, _TN, FN = tp['count'], fp['count'], tn['count'], fn['count']
    precision = TP / (TP + FP) if (TP + FP) else np.nan
    recall    = TP / (TP + FN) if (TP + FN) else np.nan

    pred_pos_count   = TP + FP
    actual_pos_count = TP + FN
    pred_pos_rate    = pred_pos_count / n
    actual_pos_rate  = actual_pos_count / n

    pred_pos_mean_x   = ((tp['mean'] * TP + fp['mean'] * FP) / pred_pos_count) if pred_pos_count else np.nan
    pred_pos_median_x = df.loc[pred == 1, x].median() if pred_pos_count else np.nan

    def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
        
        if len(a) < 2 or len(b) < 2:
            return np.nan
        
        ma, mb = np.nanmean(a), np.nanmean(b)
        va, vb = np.nanvar(a, ddof=1), np.nanvar(b, ddof=1)
        sp_num = (len(a)-1)*va + (len(b)-1)*vb
        sp_den = (len(a)+len(b)-2)
        
        if sp_den <= 0:
            return np.nan
        
        sp = sqrt(sp_num / sp_den) if sp_num > 0 else np.nan
        
        return (ma - mb) / sp if sp and not np.isnan(sp) else np.nan

    def _ks(a: np.ndarray, b: np.ndarray) -> float:
        
        if len(a) == 0 or len(b) == 0:
            return np.nan
        
        a_sorted = np.sort(a)
        b_sorted = np.sort(b)
        all_vals = np.unique(np.concatenate([a_sorted, b_sorted]))
        cdf_a = np.searchsorted(a_sorted, all_vals, side='right') / len(a_sorted)
        cdf_b = np.searchsorted(b_sorted, all_vals, side='right') / len(b_sorted)
        
        return float(np.max(np.abs(cdf_a - cdf_b)))

    tp_x = df.loc[m_tp, x].to_numpy()
    fp_x = df.loc[m_fp, x].to_numpy()
    tp_fp_cohen_d = _cohen_d(tp_x, fp_x)
    tp_fp_ks = _ks(tp_x, fp_x)

    data = pd.DataFrame.from_records([{
        **(id_cols or {}),
        'pred_pos_rate_pct': round(float(pred_pos_rate) * 100.0, 1),
        'actual_pos_rate_pct': round(float(actual_pos_rate) * 100.0, 1),
        'precision_pct': round(float(precision) * 100.0, 1),
        'recall_pct': round(float(recall) * 100.0, 1),
        'tp_x_mean': float(tp['mean']),
        'fp_x_mean': float(fp['mean']),
        'tp_x_median': float(tp['median']),
        'fp_x_median': float(fp['median']),
        'pred_pos_count': int(pred_pos_count),
        'pred_pos_x_mean': float(pred_pos_mean_x),
        'pred_pos_x_median': float(pred_pos_median_x),
        'tp_count': int(tp['count']),
        'fp_count': int(fp['count']),
        'tp_fp_cohen_d': float(tp_fp_cohen_d),
        'tp_fp_ks': float(tp_fp_ks),
        'x_name': x,
        'n_kept': int(n),
    }])
    
    return data
