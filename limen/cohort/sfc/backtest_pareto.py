from typing import Any, cast
from typing import TypeGuard

import numpy as np
import numpy.typing as npt
import pandas as pd


def _is_integral(value: Any) -> TypeGuard[int | np.integer[Any]]:

    '''
    Check whether a permutation id value is a Python or numpy integer.

    Args:
        value (Any): Candidate permutation id value

    Returns:
        TypeGuard[int | np.integer[Any]]: True if value is an integral number
    '''

    return isinstance(value, (int, np.integer))


def _is_floating(value: Any) -> TypeGuard[float | np.floating[Any]]:

    '''
    Check whether a permutation id value is a Python or numpy float.

    Args:
        value (Any): Candidate permutation id value

    Returns:
        TypeGuard[float | np.floating[Any]]: True if value is a floating number
    '''

    return isinstance(value, (float, np.floating))


def select(context: dict[str, Any],
           *,
           target_count: object = 20,
           min_signals: object = 1,
           metric_cols: list[str] | None = None) -> list[int | str]:
    '''Return a backtest-first Pareto front capped by deterministic rank.'''

    if not isinstance(target_count, int) or target_count <= 0:
        raise ValueError('backtest_pareto target_count must be a positive integer')
    if not isinstance(min_signals, int) or min_signals < 0:
        raise ValueError('backtest_pareto min_signals must be a non-negative integer')

    default_metrics = [
        'backtest_pnl_per_bar_bps',
        'backtest_edge_bps_p95',
        'backtest_wins_per_bar',
        'backtest_drawdown_bps_p5',
        'backtest_cvar_95_pnl_bps',
    ]
    metric_cols = metric_cols or default_metrics

    results = context.get('results')
    if results is None:
        raise ValueError('backtest_pareto selector requires results.csv data in context["results"]')
    if not isinstance(results, pd.DataFrame):
        raise ValueError('backtest_pareto context["results"] must be a pandas DataFrame')

    missing = [col for col in ['id', *metric_cols] if col not in results.columns]
    if missing:
        raise ValueError(f'backtest_pareto selector input is missing required columns: {missing}')

    def coerce_id(value: Any) -> int | str:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('backtest_pareto selector returned a boolean permutation id')
        if pd.isna(value):
            raise ValueError('backtest_pareto selector returned a missing permutation id')
        if _is_integral(value):
            return int(value)
        if _is_floating(value) and float(value).is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return int(stripped)
            if not stripped:
                raise ValueError('backtest_pareto selector returned an empty permutation id')
            return stripped

        coerced = str(value).strip()
        if not coerced:
            raise ValueError('backtest_pareto selector returned an empty permutation id')
        return coerced

    guard_cols = [
        col for col in ('num_trades_test', 'confusion_tp', 'confusion_fp')
        if col in results.columns
    ]
    work = results[['id', *metric_cols, *guard_cols]].copy()
    for col in [*metric_cols, *guard_cols]:
        work[col] = pd.to_numeric(work[col], errors='coerce')

    finite_metrics = cast(npt.NDArray[np.bool_], np.isfinite(work[metric_cols].to_numpy(dtype=float)).all(axis=1))
    work = work.loc[finite_metrics]

    if min_signals > 0 and 'num_trades_test' in work.columns:
        work = work.loc[work['num_trades_test'] >= min_signals]
    elif min_signals > 0 and {'confusion_tp', 'confusion_fp'} <= set(work.columns):
        signal_count = work['confusion_tp'].fillna(0) + work['confusion_fp'].fillna(0)
        work = work.loc[signal_count >= min_signals]

    work = work.dropna(subset=metric_cols)
    if work.empty:
        return []

    values = work[metric_cols].to_numpy(dtype=float)
    keep = np.ones(len(values), dtype=bool)
    for idx, row in enumerate(values):
        if not keep[idx]:
            continue
        dominates = np.all(values >= row, axis=1) & np.any(values > row, axis=1)
        dominates[idx] = False
        if dominates.any():
            keep[idx] = False

    front = work.loc[keep].copy()
    parts: list[npt.NDArray[np.float64]] = []
    for col in metric_cols:
        values = front[col].to_numpy(dtype=float)
        lo = np.nanmin(values)
        hi = np.nanmax(values)
        if not np.isfinite(lo) or not np.isfinite(hi):
            parts.append(np.zeros(len(values), dtype=float))
        elif hi == lo:
            parts.append(np.ones(len(values), dtype=float))
        else:
            parts.append((values - lo) / (hi - lo))

    front['_selector_score'] = np.vstack(parts).mean(axis=0)
    front['_id_sort_key'] = front['id'].map(lambda value: str(coerce_id(value)))
    ranked = front.sort_values(
        by=['_selector_score', '_id_sort_key'],
        ascending=[False, True],
        kind='mergesort',
    )

    return [coerce_id(value) for value in ranked.head(target_count)['id'].tolist()]
