import math

import numpy as np


def _count_entries(pos: np.ndarray) -> int:
    if len(pos) == 0:
        return 0
    first = 1 if pos[0] == 1 else 0
    return first + int(np.sum((pos[1:] == 1) & (pos[:-1] == 0)))


def rule_based_metrics(positions: dict,
                       backtest_results: dict,
                       sharpe_std_threshold: float = 0.5,
                       sharpe_degradation_threshold: float = 0.3) -> dict:

    '''
    Compute rule-based strategy metrics across train, val, and test splits.

    Args:
        positions (dict): Per-split position arrays (0/1) keyed by 'train', 'val', 'test'
        backtest_results (dict): Per-split backtest metric dicts keyed by 'train', 'val', 'test'
        sharpe_std_threshold (float): Max sharpe_std for is_stable to be True
        sharpe_degradation_threshold (float): Max sharpe_degradation for is_stable to be True

    Returns:
        dict: Tier 1 position stats, Tier 2 backtest metrics per split, Tier 3 stability metrics
    '''

    results = {}

    for split in ('train', 'val', 'test'):
        pos = np.asarray(positions.get(split, []))
        results[f'num_trades_{split}'] = _count_entries(pos)
        results[f'position_rate_{split}'] = round(float(pos.mean()), 3) if len(pos) > 0 else 0.0

    for split in ('train', 'val', 'test'):
        for k, v in backtest_results.get(split, {}).items():
            results[f'{k}_{split}'] = v

    sharpes = [backtest_results.get(s, {}).get('sharpe_per_bar') for s in ('train', 'val', 'test')]
    drawdowns = [backtest_results.get(s, {}).get('max_drawdown_pct') for s in ('train', 'val', 'test')]

    valid_sharpes = [s for s in sharpes if s is not None and not math.isnan(s)]
    valid_drawdowns = [d for d in drawdowns if d is not None and not math.isnan(d)]

    _min_splits = 2
    sharpe_std = float(np.std(valid_sharpes)) if len(valid_sharpes) >= _min_splits else float('nan')
    drawdown_std = float(np.std(valid_drawdowns)) if len(valid_drawdowns) >= _min_splits else float('nan')

    sharpe_train, sharpe_test = sharpes[0], sharpes[2]
    if (sharpe_train is not None and sharpe_test is not None
            and not math.isnan(sharpe_train) and abs(sharpe_train) > 0):
        sharpe_degradation = (sharpe_train - sharpe_test) / abs(sharpe_train)
    else:
        sharpe_degradation = float('nan')

    results['sharpe_std'] = round(sharpe_std, 3) if not math.isnan(sharpe_std) else None
    results['drawdown_std'] = round(drawdown_std, 3) if not math.isnan(drawdown_std) else None
    results['sharpe_degradation'] = round(sharpe_degradation, 3) if not math.isnan(sharpe_degradation) else None
    results['is_stable'] = (
        not math.isnan(sharpe_std) and sharpe_std < sharpe_std_threshold
        and not math.isnan(sharpe_degradation) and sharpe_degradation < sharpe_degradation_threshold
    )

    return results
