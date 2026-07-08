from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from limen.backtest.backtest_snapshot import backtest_snapshot
from limen.metrics.rule_based_metrics import rule_based_metrics
from limen.sfd.reference_architecture.base import ReferenceModel


class RuleBasedStrategy(ReferenceModel):

    '''Rule-based strategy that applies boolean predicate logic per bar to produce long/flat positions.'''

    deterministic = True

    def __init__(self,
                 sharpe_std_threshold: float = 0.5,
                 sharpe_degradation_threshold: float = 0.3) -> None:

        super().__init__()
        self.sharpe_std_threshold = sharpe_std_threshold
        self.sharpe_degradation_threshold = sharpe_degradation_threshold

    def train(self, data: dict, **params: Any) -> 'RuleBasedStrategy':  # noqa: ARG002

        '''
        No-op training step — rule-based strategies have no learnable parameters.

        Args:
            data (dict): Ignored
            **params: Ignored

        Returns:
            RuleBasedStrategy: Self
        '''

        return self

    def predict(self, data: dict) -> dict:

        '''
        Apply boolean logic tree to test split and return per-bar position signals.

        Args:
            data (dict): Data dict with 'test' DataFrame and 'strategy' config

        Returns:
            dict: {'_preds': np.ndarray} of 0/1 integer positions
        '''

        pos = self._apply_logic(data['test'], data['strategy']).fill_null(False).to_numpy().astype(int)
        return {'_preds': pos}

    def evaluate(self, data: dict, inline_metrics: bool = True) -> dict:  # noqa: ARG002

        '''
        Evaluate strategy across all splits and return rule-based metrics.

        Args:
            data (dict): Data dict with 'train', 'val', 'test' DataFrames and 'strategy' config
            inline_metrics (bool): Unused — included for interface compatibility

        Returns:
            dict: Tier 1 position stats, Tier 2 backtest metrics per split,
                Tier 3 stability metrics, and '_preds' key
        '''

        positions: dict[str, np.ndarray] = {}
        backtest_results: dict[str, dict] = {}
        strategy = data['strategy']
        cond_index = {c['id']: c for c in strategy['conditions']}

        cost_kwargs = self._cost_kwargs(data)
        for split in ('train', 'val', 'test'):
            pos = self._resolve(cond_index[strategy['entry']], cond_index, data[split]).fill_null(False).to_numpy().astype(int)
            positions[split] = pos
            backtest_results[split] = self._backtest_split(data[split], pos, cost_kwargs)

        results = rule_based_metrics(
            positions,
            backtest_results,
            sharpe_std_threshold=self.sharpe_std_threshold,
            sharpe_degradation_threshold=self.sharpe_degradation_threshold,
        )
        results['_preds'] = positions['test']
        return results

    def _apply_logic(self, df: pl.DataFrame, strategy: dict) -> pl.Series:
        cond_index = {c['id']: c for c in strategy['conditions']}
        return self._resolve(cond_index[strategy['entry']], cond_index, df)

    def _resolve(self, condition: dict, cond_index: dict, df: pl.DataFrame) -> pl.Series:
        if 'type' in condition:
            return df[condition['id']]
        operator = condition['operator']
        if operator not in ('and', 'or', 'not'):
            raise ValueError(f'Unknown logical operator: {operator!r}')
        operands = [self._resolve(cond_index[op_id], cond_index, df) for op_id in condition.get('operands', [])]
        if not operands:
            raise ValueError(f'Compound condition {condition.get("id")!r} has no operands')
        if operator == 'not':
            if len(operands) != 1:
                raise ValueError(f'NOT operator requires exactly 1 operand, got {len(operands)}')
            return ~operands[0]
        result = operands[0]
        for s in operands[1:]:
            result = result & s if operator == 'and' else result | s
        return result

    def _backtest_split(self, df: pl.DataFrame, positions: np.ndarray, cost_kwargs: dict) -> dict:
        if 'open' not in df.columns or 'close' not in df.columns:
            return {}
        open_arr = df['open'].to_numpy().astype(float)
        close_arr = df['close'].to_numpy().astype(float)
        bt_input_data = {
            'predictions': positions,
            'open': open_arr,
            'close': close_arr,
            'price_change': close_arr - open_arr,
        }
        if 'datetime' in df.columns:
            bt_input_data['datetime'] = df['datetime'].to_numpy()

        bt_input = pd.DataFrame(bt_input_data)
        bt_result = backtest_snapshot(bt_input, execution_lag_bars=1, **cost_kwargs)
        if bt_result.empty:
            return {}
        return bt_result.iloc[0].to_dict()


def rule_based(data: dict,
               sharpe_std_threshold: float = 0.5,
               sharpe_degradation_threshold: float = 0.3) -> dict:

    '''
    Apply a rule-based strategy to the given data and return evaluation metrics.

    Args:
        data (dict): Data dict with 'train', 'val', 'test' DataFrames and 'strategy' config
        sharpe_std_threshold (float): Max sharpe_std for is_stable to be True
        sharpe_degradation_threshold (float): Max sharpe_degradation for is_stable to be True

    Returns:
        dict: Rule-based metrics with Tier 1, Tier 2, Tier 3 keys and '_preds'
    '''

    model = RuleBasedStrategy(
        sharpe_std_threshold=sharpe_std_threshold,
        sharpe_degradation_threshold=sharpe_degradation_threshold,
    )
    _ = model.train(data)
    return model.evaluate(data)
