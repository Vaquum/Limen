from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SelectorContext = dict[str, Any]
Selector = Callable[..., list[int | str]]

ID_COL = 'id'
MIN_DIVERSE_METRIC_COLS = 2

BACKTEST_PARETO_METRICS = (
    'backtest_trade_pnl_net_bps_p50',
    'backtest_edge_per_signal_bps_p50',
    'backtest_return_on_exposure_p50',
    'backtest_drawdown_depth_bps_p50',
    'backtest_cvar_95_return_bps',
)

LEGACY_DIVERSITY_METRIC_COLS = (
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
)

INLINE_CONFUSION_METRIC_COLS = (
    'precision',
    'recall',
    'auc',
    'accuracy',
    'confusion_precision',
    'confusion_recall',
    'confusion_tp_mean_return_pct',
    'confusion_fp_mean_return_pct',
)


def select_all(context: SelectorContext) -> list[int | str]:
    '''Return every available permutation id.'''

    return _sort_ids(context['available_permutation_ids'])


def select_top_n(context: SelectorContext,
                 *,
                 column: str,
                 n: int,
                 ascending: bool = False) -> list[int | str]:
    '''Return the top n ids from results.csv ordered by one numeric column.'''

    _require_positive_int(n, 'n')
    results = _results(context)
    _require_columns(results, [ID_COL, column])

    work = _numeric_frame(results, [column]).dropna(subset=[column])
    if work.empty:
        return []

    work = _with_id_sort_key(work)
    ranked = work.sort_values(
        by=[column, '_id_sort_key'],
        ascending=[ascending, True],
        kind='mergesort',
    )

    return _ids_from_frame(ranked.head(n))


def select_backtest_pareto(context: SelectorContext,
                           *,
                           target_count: int = 20,
                           min_signals: int = 1,
                           metric_cols: list[str] | None = None) -> list[int | str]:
    '''Return a backtest-first Pareto front capped by deterministic rank.'''

    _require_positive_int(target_count, 'target_count')
    _require_non_negative_int(min_signals, 'min_signals')

    results = _results(context)
    metric_cols = metric_cols or list(BACKTEST_PARETO_METRICS)
    _require_columns(results, [ID_COL, *metric_cols])

    guard_cols = [
        col for col in ('num_trades_test', 'confusion_tp', 'confusion_fp')
        if col in results.columns
    ]
    work = _numeric_frame(results, [*metric_cols, *guard_cols])
    work = _apply_signal_filter(work, min_signals)
    work = work.dropna(subset=metric_cols)
    if work.empty:
        return []

    front = work.loc[_pareto_mask(work, metric_cols)].copy()
    ranked = _rank_by_normalized_score(front, metric_cols)

    return _ids_from_frame(ranked.head(target_count))


def select_diverse_metrics(context: SelectorContext,
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

    _require_positive_int(target_count, 'target_count')
    _require_positive_int(n_clusters, 'n_clusters')
    if n_components is not None:
        _require_positive_int(n_components, 'n_components')
    if iqr_multiplier < 0:
        raise ValueError('iqr_multiplier must be >= 0')

    results = _results(context)
    metric_cols = metric_cols or _default_diverse_metric_cols(results)
    _require_columns(results, [ID_COL, *metric_cols])

    work = _numeric_frame(results, metric_cols).dropna(subset=metric_cols)
    if work.empty:
        return []

    filtered = _iqr_filter(work, metric_cols, iqr_multiplier)
    if not filtered.empty:
        work = filtered

    if len(work) <= target_count:
        return _ids_from_frame(work)

    selected_positions = _pca_kmeans_medoids(
        work,
        metric_cols,
        n_components=n_components,
        n_clusters=n_clusters,
        random_state=random_state,
    )

    if len(selected_positions) < target_count:
        ranked_positions = _rank_positions(work, metric_cols)
        for idx in ranked_positions:
            if len(selected_positions) >= target_count:
                break
            if idx not in selected_positions:
                selected_positions.append(idx)

    selected_positions = list(dict.fromkeys(selected_positions))[:target_count]
    return _ids_from_frame(work.iloc[selected_positions])


BUILTIN_SELECTORS: dict[str, Selector] = {
    'all': select_all,
    'top_n': select_top_n,
    'backtest_pareto': select_backtest_pareto,
    'diverse_metrics': select_diverse_metrics,
}


def _results(context: SelectorContext) -> pd.DataFrame:
    results = context.get('results')
    if results is None:
        raise ValueError('selector requires results.csv data in context["results"]')
    if not isinstance(results, pd.DataFrame):
        raise ValueError('context["results"] must be a pandas DataFrame')
    return results.copy()


def _sort_ids(values: Any) -> list[int | str]:
    return sorted(
        values,
        key=lambda value: (0, value) if isinstance(value, int) else (1, str(value)),
    )


def _ids_from_frame(df: pd.DataFrame) -> list[int | str]:
    return [_coerce_id(value) for value in df[ID_COL].tolist()]


def _coerce_id(value: Any) -> int | str:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('selector returned a boolean permutation id')
    if pd.isna(value):
        raise ValueError('selector returned a missing permutation id')
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        if not stripped:
            raise ValueError('selector returned an empty permutation id')
        return stripped

    coerced = str(value).strip()
    if not coerced:
        raise ValueError('selector returned an empty permutation id')
    return coerced


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f'selector input is missing required columns: {missing}')


def _require_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def _require_non_negative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer')


def _numeric_frame(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    work = df[[ID_COL, *metric_cols]].copy()
    for col in metric_cols:
        work[col] = pd.to_numeric(work[col], errors='coerce')
    return work


def _with_id_sort_key(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work['_id_sort_key'] = work[ID_COL].map(lambda value: str(_coerce_id(value)))
    return work


def _apply_signal_filter(df: pd.DataFrame, min_signals: int) -> pd.DataFrame:
    if min_signals == 0:
        return df

    work = df.copy()
    if 'num_trades_test' in work.columns:
        signal_count = pd.to_numeric(work['num_trades_test'], errors='coerce')
    elif {'confusion_tp', 'confusion_fp'} <= set(work.columns):
        tp = pd.to_numeric(work['confusion_tp'], errors='coerce').fillna(0)
        fp = pd.to_numeric(work['confusion_fp'], errors='coerce').fillna(0)
        signal_count = tp + fp
    else:
        return work

    return work.loc[signal_count >= min_signals]


def _pareto_mask(df: pd.DataFrame, metric_cols: list[str]) -> np.ndarray:
    values = df[metric_cols].to_numpy(dtype=float)
    keep = np.ones(len(values), dtype=bool)

    for idx, row in enumerate(values):
        if not keep[idx]:
            continue
        dominates = np.all(values >= row, axis=1) & np.any(values > row, axis=1)
        dominates[idx] = False
        if dominates.any():
            keep[idx] = False

    return keep


def _rank_by_normalized_score(df: pd.DataFrame,
                              metric_cols: list[str]) -> pd.DataFrame:
    work = df.copy()
    work['_selector_score'] = _normalized_score(work, metric_cols)
    work = _with_id_sort_key(work)
    return work.sort_values(
        by=['_selector_score', '_id_sort_key'],
        ascending=[False, True],
        kind='mergesort',
    )


def _rank_positions(df: pd.DataFrame, metric_cols: list[str]) -> list[int]:
    scores = _normalized_score(df, metric_cols)
    return list(np.argsort(-scores))


def _normalized_score(df: pd.DataFrame, metric_cols: list[str]) -> np.ndarray:
    parts = []
    for col in metric_cols:
        values = df[col].to_numpy(dtype=float)
        lo = np.nanmin(values)
        hi = np.nanmax(values)
        if not np.isfinite(lo) or not np.isfinite(hi):
            parts.append(np.zeros(len(values), dtype=float))
        elif hi == lo:
            parts.append(np.ones(len(values), dtype=float))
        else:
            parts.append((values - lo) / (hi - lo))
    return np.vstack(parts).mean(axis=0)


def _default_diverse_metric_cols(df: pd.DataFrame) -> list[str]:
    for cols in (
        BACKTEST_PARETO_METRICS,
        LEGACY_DIVERSITY_METRIC_COLS,
        INLINE_CONFUSION_METRIC_COLS,
    ):
        present = [col for col in cols if col in df.columns]
        if len(present) >= MIN_DIVERSE_METRIC_COLS:
            return present
    raise ValueError('diverse_metrics requires at least two numeric metric columns')


def _iqr_filter(df: pd.DataFrame,
                metric_cols: list[str],
                iqr_multiplier: float) -> pd.DataFrame:
    work = df.copy()
    for col in metric_cols:
        q1 = work[col].quantile(0.25)
        q3 = work[col].quantile(0.75)
        if pd.isna(q1) or pd.isna(q3) or q1 == q3:
            continue
        iqr = q3 - q1
        lo = q1 - iqr_multiplier * iqr
        hi = q3 + iqr_multiplier * iqr
        work = work.loc[(work[col] >= lo) & (work[col] <= hi)]
    return work


def _pca_kmeans_medoids(df: pd.DataFrame,
                        metric_cols: list[str],
                        *,
                        n_components: int | None,
                        n_clusters: int,
                        random_state: int) -> list[int]:
    values = df[metric_cols].to_numpy(dtype=float)
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
        return [int(np.argmin(distances))]

    kmeans = KMeans(
        n_clusters=actual_clusters,
        random_state=random_state,
        n_init='auto',
    )
    labels = kmeans.fit_predict(projected)

    selected: list[int] = []
    for cluster_id in range(actual_clusters):
        mask = labels == cluster_id
        idxs = np.nonzero(mask)[0]
        if idxs.size == 0:
            continue
        cluster_points = projected[mask]
        center = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_points - center, axis=1)
        selected.append(int(idxs[int(np.argmin(distances))]))

    return selected
