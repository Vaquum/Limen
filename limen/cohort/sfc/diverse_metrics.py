from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def select(context: dict[str, Any],
           *,
           target_count: int = 20,
           metric_cols: list[str] | None = None,
           iqr_multiplier: float = 3.0,
           n_components: int | None = None,
           n_clusters: int = 8,
           random_state: int = 42) -> list[int | str]:
    '''
    Return a metric-diverse set of ids using PCA/KMeans medoid selection.

    The selector prefers backtest metrics when they exist. If not, it falls
    back to the legacy confusion-metric surface, then inline confusion
    metrics from results.csv.
    '''

    if not isinstance(target_count, int) or target_count <= 0:
        raise ValueError('diverse_metrics target_count must be a positive integer')
    if not isinstance(n_clusters, int) or n_clusters <= 0:
        raise ValueError('diverse_metrics n_clusters must be a positive integer')
    if n_components is not None and (
        not isinstance(n_components, int) or n_components <= 0
    ):
        raise ValueError('diverse_metrics n_components must be a positive integer')
    if iqr_multiplier < 0:
        raise ValueError('diverse_metrics iqr_multiplier must be >= 0')

    min_default_metric_cols = 2
    results = context.get('results')
    if results is None:
        raise ValueError('diverse_metrics selector requires results.csv data in context["results"]')
    if not isinstance(results, pd.DataFrame):
        raise ValueError('diverse_metrics context["results"] must be a pandas DataFrame')

    metric_groups = [
        [
            'backtest_trade_pnl_net_bps_p50',
            'backtest_edge_per_signal_bps_p50',
            'backtest_return_on_exposure_p50',
            'backtest_drawdown_depth_bps_p50',
            'backtest_cvar_95_return_bps',
        ],
        [
            'pred_pos_rate_pct',
            'actual_pos_rate_pct',
            'precision_pct',
            'recall_pct',
            'tp_x_mean',
            'fp_x_mean',
            'tp_x_median',
            'fp_x_median',
            'pred_pos_count',
            'pred_pos_x_mean',
            'pred_pos_x_median',
            'tp_count',
            'fp_count',
            'tp_fp_cohen_d',
            'tp_fp_ks',
        ],
        [
            'precision',
            'recall',
            'auc',
            'accuracy',
            'confusion_precision',
            'confusion_recall',
            'confusion_tp_mean_return_pct',
            'confusion_fp_mean_return_pct',
        ],
    ]
    if metric_cols is None:
        for group in metric_groups:
            present = [col for col in group if col in results.columns]
            if len(present) >= min_default_metric_cols:
                metric_cols = present
                break
        if metric_cols is None:
            raise ValueError('diverse_metrics requires at least two numeric metric columns')

    missing = [col for col in ['id', *metric_cols] if col not in results.columns]
    if missing:
        raise ValueError(f'diverse_metrics selector input is missing required columns: {missing}')

    def coerce_id(value: Any) -> int | str:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('diverse_metrics selector returned a boolean permutation id')
        if pd.isna(value):
            raise ValueError('diverse_metrics selector returned a missing permutation id')
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return int(stripped)
            if not stripped:
                raise ValueError('diverse_metrics selector returned an empty permutation id')
            return stripped

        coerced = str(value).strip()
        if not coerced:
            raise ValueError('diverse_metrics selector returned an empty permutation id')
        return coerced

    work = results[['id', *metric_cols]].copy()
    for col in metric_cols:
        work[col] = pd.to_numeric(work[col], errors='coerce')
    work = work.dropna(subset=metric_cols)
    if work.empty:
        return []

    filtered = work.copy()
    for col in metric_cols:
        q1 = filtered[col].quantile(0.25)
        q3 = filtered[col].quantile(0.75)
        if pd.isna(q1) or pd.isna(q3) or q1 == q3:
            continue
        iqr = q3 - q1
        lo = q1 - iqr_multiplier * iqr
        hi = q3 + iqr_multiplier * iqr
        filtered = filtered.loc[(filtered[col] >= lo) & (filtered[col] <= hi)]
    if not filtered.empty:
        work = filtered

    if len(work) <= target_count:
        return [coerce_id(value) for value in work['id'].tolist()]

    values = work[metric_cols].to_numpy(dtype=float)
    n_samples, n_features = values.shape
    actual_clusters = min(n_clusters, n_samples)
    actual_components = (
        None
        if n_components is None
        else min(n_components, n_samples, n_features)
    )

    scaled = StandardScaler().fit_transform(values)
    projected = PCA(
        n_components=actual_components,
        random_state=random_state,
    ).fit_transform(scaled)

    if actual_clusters == 1:
        center = projected.mean(axis=0)
        distances = np.linalg.norm(projected - center, axis=1)
        selected_positions = [int(np.argmin(distances))]
    else:
        kmeans = KMeans(
            n_clusters=actual_clusters,
            random_state=random_state,
            n_init='auto',
        )
        labels = kmeans.fit_predict(projected)

        selected_positions: list[int] = []
        for cluster_id in range(actual_clusters):
            mask = labels == cluster_id
            idxs = np.nonzero(mask)[0]
            if idxs.size == 0:
                continue
            cluster_points = projected[mask]
            center = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_points - center, axis=1)
            selected_positions.append(int(idxs[int(np.argmin(distances))]))

    if len(selected_positions) < target_count:
        parts = []
        for col in metric_cols:
            col_values = work[col].to_numpy(dtype=float)
            lo = np.nanmin(col_values)
            hi = np.nanmax(col_values)
            if not np.isfinite(lo) or not np.isfinite(hi):
                parts.append(np.zeros(len(col_values), dtype=float))
            elif hi == lo:
                parts.append(np.ones(len(col_values), dtype=float))
            else:
                parts.append((col_values - lo) / (hi - lo))

        scores = np.vstack(parts).mean(axis=0)
        for ranked_idx in np.argsort(-scores):
            if len(selected_positions) >= target_count:
                break
            idx = int(ranked_idx)
            if idx not in selected_positions:
                selected_positions.append(idx)

    selected_positions = list(dict.fromkeys(selected_positions))[:target_count]
    return [
        coerce_id(value)
        for value in work.iloc[selected_positions]['id'].tolist()
    ]
