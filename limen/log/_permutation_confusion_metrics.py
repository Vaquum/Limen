import polars as pl
import numpy as np
import numpy.typing as npt
from typing import Any, cast
from collections.abc import Sequence
from math import sqrt


class ConfusionNotApplicableError(ValueError):
    '''Recorded outputs carry no binary classification interpretation.'''


def _declared_output_modes(manifest: object, alignment: object, round_id: int) -> set[str]:
    modes = {mode for mode in (getattr(getattr(manifest, 'backtest_config', None), 'prediction_mode', None),) if isinstance(mode, str)}
    rows: list[object] = cast(list[object], alignment) if isinstance(alignment, list) else []
    node: object = rows[round_id] if -len(rows) <= round_id < len(rows) else None
    for key in ('trade_contract', 'policy', 'prediction_mode'):
        mapping = cast(dict[object, object], node) if isinstance(node, dict) else None
        node = mapping.get(key) if mapping is not None else None
    return modes | {node} if isinstance(node, str) else modes


def confusion_mean_return_pct(pred: pl.Series | npt.ArrayLike,
                               actual: pl.Series | npt.ArrayLike,
                               open_px: pl.Series,
                               price_change: pl.Series,
                               *,
                               execution_lag_bars: int = 1) -> dict[str, float]:
    '''
    Compute execution-aligned mean return percentage for each confusion quadrant.

    Args:
        pred (pl.Series | np.ndarray): Binary predictions
        actual (pl.Series | np.ndarray): Binary actuals
        open_px (pl.Series): Open prices for the feature-bar rows
        price_change (pl.Series): Close - open for the feature-bar rows
        execution_lag_bars (int): Number of rows forward to evaluate on

    Returns:
        dict[str, float]: Mean return percentages for TP/FP/TN/FN
    '''

    if execution_lag_bars < 0:
        raise ValueError('permutation_confusion_metrics execution_lag_bars must be >= 0')

    pred = pl.Series(np.asarray(pred)).cast(pl.Float64, strict=False)
    actual = pl.Series(np.asarray(actual)).cast(pl.Float64, strict=False)
    open_px = pl.Series(np.asarray(open_px)).cast(pl.Float64, strict=False)
    price_change = pl.Series(np.asarray(price_change)).cast(pl.Float64, strict=False)

    if execution_lag_bars:
        open_px = open_px.shift(-execution_lag_bars)
        price_change = price_change.shift(-execution_lag_bars)

    pred_arr = pred.to_numpy()
    actual_arr = actual.to_numpy()
    open_arr = open_px.to_numpy()
    change_arr = price_change.to_numpy()

    return_pct = np.divide(change_arr, open_arr, out=cast(npt.NDArray[np.float64], np.full(open_arr.shape, np.nan, dtype=np.float64)), where=open_arr != 0) * 100.0
    valid = (
        ~np.isnan(pred_arr) &
        ~np.isnan(actual_arr) &
        ~np.isnan(open_arr) &
        ~np.isnan(change_arr) &
        (open_arr != 0) &
        np.isfinite(return_pct)
    )

    pred_arr, actual_arr = pred_arr[valid], actual_arr[valid]
    if ((pred_arr != 0) & (pred_arr != 1)).any() or ((actual_arr != 0) & (actual_arr != 1)).any():
        raise ValueError('confusion_mean_return_pct requires binary predictions and actuals')
    pred_arr = pred_arr.astype(int)
    actual_arr = actual_arr.astype(int)
    return_pct = return_pct[valid]

    def _mean(mask: npt.NDArray[np.bool_]) -> float:
        values = return_pct[mask]
        return float(float(values.mean())) if values.size else np.nan

    return {
        'tp_mean_return_pct': _mean((pred_arr == 1) & (actual_arr == 1)),
        'fp_mean_return_pct': _mean((pred_arr == 1) & (actual_arr == 0)),
        'tn_mean_return_pct': _mean((pred_arr == 0) & (actual_arr == 0)),
        'fn_mean_return_pct': _mean((pred_arr == 0) & (actual_arr == 1)),
    }


def permutation_confusion_metrics(self: Any,
                                  x: str,
                                  round_id: int,
                                  *,
                                  pred_col: str = 'predictions',
                                  actual_col: str = 'actuals',
                                  proba_col: str | None = None,
                                  threshold: float = 0.5,
                                  outlier_quantiles: Sequence[float] = (0.01, 0.99),
                                  outlier_mode: str = 'filter',
                                  id_cols: dict[str, Any] | None = None) -> pl.DataFrame:
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
        pl.DataFrame: One-row table with columns 'x_name', 'n_kept', 'pred_pos_rate_pct',
                      'actual_pos_rate_pct', 'precision_pct', 'recall_pct', 'pred_pos_count',
                      'tp_count', 'fp_count', 'tp_x_mean', 'tp_x_median', 'fp_x_mean', 'fp_x_median',
                      'pred_pos_x_mean', 'pred_pos_x_median', 'tp_fp_cohen_d', 'tp_fp_ks',
                      'tp_mean_return_pct', 'fp_mean_return_pct', 'tn_mean_return_pct',
                      'fn_mean_return_pct'
    '''

    df = self.permutation_prediction_performance(round_id)

    # Optional: binarize from probabilities
    if proba_col is not None:
        if proba_col not in df.columns:
            raise ValueError(f'permutation_confusion_metrics proba_col "{proba_col}" not found')

        df = df.with_columns((pl.col(proba_col).cast(pl.Float64).fill_nan(None) >= float(threshold)).fill_null(False).cast(pl.Int64).alias(pred_col))

    # Validate required columns
    for col in (pred_col, actual_col, x, 'open', 'price_change'):
        if col not in df.columns:
            raise ValueError(f'permutation_confusion_metrics column "{col}" not found')

    foreign = _declared_output_modes(getattr(self, 'manifest', None), getattr(self, '_alignment', None), round_id) - {'binary'}
    if foreign:
        raise ConfusionNotApplicableError(f'permutation_confusion_metrics inapplicable to {sorted(foreign)} outputs')

    pred_num = df[pred_col].cast(pl.Float64, strict=False)
    actual_num = df[actual_col].cast(pl.Float64, strict=False)
    eligible = pred_num.is_not_null() & pred_num.is_not_nan() & actual_num.is_not_null() & actual_num.is_not_nan()
    paired = pl.concat([pred_num.filter(eligible), actual_num.filter(eligible)])
    if paired.is_empty() or ((paired != 0) & (paired != 1)).any():
        raise ConfusionNotApplicableError('permutation_confusion_metrics requires binary predictions and actuals')

    mean_return_pct = confusion_mean_return_pct(
        df[pred_col],
        df[actual_col],
        df['open'],
        df['price_change'],
    )
    df = df.filter(eligible).with_columns(
        pl.col(pred_col).cast(pl.Int64), pl.col(actual_col).cast(pl.Int64), pl.col(x).fill_nan(None),
    )

    q_lo, q_hi = (df[x].quantile(q, interpolation='linear') for q in outlier_quantiles)
    if outlier_mode == 'filter':
        df = df.filter(pl.col(x).is_not_null() & pl.col(x).is_between(q_lo, q_hi))

    elif outlier_mode == 'winsor':
        df = df.with_columns(pl.col(x).clip(q_lo, q_hi))

    else:
        raise ValueError('permutation_confusion_metrics outlier_mode must be "filter" or "winsor"')

    n = len(df)
    if n == 0:
        raise ValueError('permutation_confusion_metrics no rows remain after outlier handling')

    pred = df[pred_col]
    act = df[actual_col]
    m_tp = (pred == 1) & (act == 1)
    m_fp = (pred == 1) & (act == 0)
    m_tn = (pred == 0) & (act == 0)
    m_fn = (pred == 0) & (act == 1)

    def _stats(mask: pl.Series) -> dict[str, float]:
        s = df.filter(mask)[x]
        k = int(mask.sum())
        return {
            'count': k,
            'mean': float(value) if (value := s.mean()) is not None else np.nan,
            'median': float(value) if (value := s.median()) is not None else np.nan,
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
    pred_pos_median_x = float(value) if (value := df.filter(pred == 1)[x].median()) is not None else np.nan

    def _cohen_d(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:

        MIN_SAMPLES_FOR_COHEN_D = 2
        if len(a) < MIN_SAMPLES_FOR_COHEN_D or len(b) < MIN_SAMPLES_FOR_COHEN_D:
            return np.nan

        ma, mb = np.nanmean(a), np.nanmean(b)
        va, vb = np.nanvar(a, ddof=1), np.nanvar(b, ddof=1)
        sp_num = (len(a)-1)*va + (len(b)-1)*vb
        sp_den = (len(a)+len(b)-MIN_SAMPLES_FOR_COHEN_D)

        if sp_den <= 0:
            return np.nan

        sp = sqrt(sp_num / sp_den) if sp_num > 0 else np.nan

        return float((ma - mb) / sp) if sp and not np.isnan(sp) else np.nan

    def _ks(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:

        if len(a) == 0 or len(b) == 0:
            return np.nan

        a_sorted = np.sort(a)
        b_sorted = np.sort(b)
        all_vals = np.unique(np.concatenate([a_sorted, b_sorted]))
        cdf_a = np.searchsorted(a_sorted, all_vals, side='right') / len(a_sorted)
        cdf_b = np.searchsorted(b_sorted, all_vals, side='right') / len(b_sorted)

        return float(np.max(np.abs(cdf_a - cdf_b)))

    tp_x = df.filter(m_tp)[x].to_numpy()
    fp_x = df.filter(m_fp)[x].to_numpy()
    tp_fp_cohen_d = _cohen_d(tp_x, fp_x)
    tp_fp_ks = _ks(tp_x, fp_x)

    return pl.DataFrame([{
        **(id_cols or {}),
        'pred_pos_rate_pct': float(float(pred_pos_rate) * 100.0),
        'actual_pos_rate_pct': float(float(actual_pos_rate) * 100.0),
        'precision_pct': float(float(precision) * 100.0),
        'recall_pct': float(float(recall) * 100.0),
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
        **mean_return_pct,
    }])
