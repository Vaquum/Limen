from itertools import combinations
from math import comb

import numpy as np
import numpy.typing as npt

__all__ = ['probability_of_backtest_overfitting']

_MINIMUM_TRIALS = 2
_MINIMUM_BLOCKS = 2
_MINIMUM_BLOCK_BARS = 2
_MATRIX_DIMENSIONS = 2
_MEDIAN_RANK = 0.5


def _sharpe_ratios(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        spreads = np.std(values, axis=1, ddof=1)
        if not np.isfinite(spreads).all() or np.any(spreads <= 0):
            raise ValueError('PBO requires positive finite trial variance in every half sample')
        ratios = np.mean(values, axis=1) / spreads
    if not np.isfinite(ratios).all():
        raise ValueError('PBO half-sample Sharpe ratios must be finite')
    return ratios


def probability_of_backtest_overfitting(
    returns_matrix: npt.NDArray[np.float64], *, n_blocks: int,
) -> float:
    '''Return CSCV logit mass at or below zero using per-bar Sharpe ranks.

    https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf, algorithm 2.3.
    Rows are trials, columns synchronous bars. Equal contiguous blocks need
    at least two bars. In-sample ties select the first trial; out-of-sample
    ties receive their average ascending rank. Median ties count as overfit.
    '''
    values = np.asarray(returns_matrix, dtype=np.float64)
    if values.ndim != _MATRIX_DIMENSIONS or values.shape[0] < _MINIMUM_TRIALS:
        raise ValueError('PBO requires a trial-by-bar matrix with at least two trials')
    if type(n_blocks) is not int or n_blocks < _MINIMUM_BLOCKS or n_blocks % 2:
        raise ValueError('PBO n_blocks must be an even integer of at least two')
    n_trials, n_bars = values.shape
    if n_bars % n_blocks or n_bars // n_blocks < _MINIMUM_BLOCK_BARS:
        raise ValueError('PBO requires equal contiguous blocks with at least two bars each')
    if not np.isfinite(values).all():
        raise ValueError('PBO returns must be finite')
    blocks = values.reshape(n_trials, n_blocks, -1)
    overfit = 0
    for selected in combinations(range(n_blocks), n_blocks // 2):
        held_out = tuple(block for block in range(n_blocks) if block not in selected)
        in_sample = _sharpe_ratios(blocks[:, selected, :].reshape(n_trials, -1))
        out_of_sample = _sharpe_ratios(blocks[:, held_out, :].reshape(n_trials, -1))
        winner = int(np.argmax(in_sample))
        selected_score = out_of_sample[winner]
        rank = (float(np.count_nonzero(out_of_sample < selected_score))
                + (float(np.count_nonzero(out_of_sample == selected_score)) + 1) / 2)
        overfit += rank / (n_trials + 1) <= _MEDIAN_RANK
    return overfit / comb(n_blocks, n_blocks // 2)
