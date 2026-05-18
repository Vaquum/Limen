from typing import Any

import numpy as np


_SPLITS = ('train', 'val', 'test')
_MIN_SPLITS = 2


def _count_entries(pos: np.ndarray) -> int:
    if len(pos) == 0:
        return 0
    return int(pos[0] == 1) + int(np.sum(np.diff(pos) > 0))


def _round_or_none(x: float, decimals: int = 3) -> float | None:
    return round(x, decimals) if not np.isnan(x) else None


def rule_based_metrics(positions: dict,
                       backtest_results: dict,
                       sharpe_std_threshold: float = 0.5,
                       sharpe_degradation_threshold: float = 0.3) -> dict:

    '''
    Compute rule-based strategy metrics across train, val, and test splits.

    Args:
        positions (dict): Per-split position arrays (0/1) keyed by 'train', 'val', 'test'
        backtest_results (dict): Per-split backtest metric dicts keyed by 'train', 'val', 'test'
        sharpe_std_threshold (float): Legacy parameter retained for call compatibility
        sharpe_degradation_threshold (float): Legacy parameter retained for call compatibility

    Returns:
        dict: Tier 1 position stats, Tier 2 backtest metrics per split, Tier 3 stability metrics
    '''

    _ = (sharpe_std_threshold, sharpe_degradation_threshold)

    results: dict[str, Any] = {}
    split_bt: dict[str, dict] = {}

    for split in _SPLITS:
        pos = np.asarray(positions.get(split, []))
        results[f'num_trades_{split}'] = _count_entries(pos)
        results[f'position_rate_{split}'] = round(float(pos.mean()), 3) if len(pos) > 0 else 0.0
        bt = backtest_results.get(split, {})
        split_bt[split] = bt
        for k, v in bt.items():
            results[f'{k}_{split}'] = v

    drawdowns = [split_bt[s].get('drawdown_depth_bps_p5') for s in _SPLITS]

    valid_drawdowns = [d for d in drawdowns if d is not None and not np.isnan(d)]

    drawdown_std_bps = float(np.std(valid_drawdowns, ddof=1)) if len(valid_drawdowns) >= _MIN_SPLITS else float('nan')

    results['drawdown_std_bps'] = _round_or_none(drawdown_std_bps)
    results['is_stable'] = False

    return results
