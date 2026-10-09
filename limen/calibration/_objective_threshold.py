from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from limen.backtest.trade_contract import finite_number


def validate_probabilities(values: object) -> npt.NDArray[np.float64]:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 1 or not probabilities.size:
        raise ValueError('Objective requires nonempty one-dimensional validation probabilities')
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Objective validation probabilities must be finite and in [0, 1]')
    return probabilities


def select_threshold(y_val: object, val_proba: object,
                     thresholds: npt.NDArray[np.float64],
                     metric: Callable[[object, object], float],
                     maximize: bool) -> tuple[float, float]:
    probabilities = validate_probabilities(val_proba)
    candidates = [2.0, *np.unique(thresholds)[::-1].tolist()]
    best_threshold = candidates[0]
    best_score = finite_number(float(metric(y_val, (probabilities >= best_threshold).astype(np.int8))), 'objective score')
    for threshold in candidates[1:]:
        score = finite_number(float(metric(y_val, (probabilities >= threshold).astype(np.int8))), 'objective score')
        if (maximize and score > best_score) or (not maximize and score < best_score):
            best_threshold, best_score = threshold, score
    return float(best_threshold), best_score


__all__ = ['select_threshold', 'validate_probabilities']
