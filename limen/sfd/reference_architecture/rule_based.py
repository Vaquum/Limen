from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from typing_extensions import override

from limen.sfd.reference_architecture._backtest_evaluation import evaluate_prices as _evaluate_prices
from limen.sfd.reference_architecture._backtest_evaluation import record_execution as _record_execution
from limen.backtest.long_flat_strategy import ExecutionResult
from limen.metrics.rule_based_metrics import rule_based_metrics
from limen.sfd.reference_architecture.base import ReferenceModel

BPS_PER_UNIT = 10_000.0
BPS_DECIMALS = 1


def _compounded_trade_pnl_summary(
    result: ExecutionResult,
    notional_rate: float,
) -> tuple[float, int]:
    in_market = np.asarray(result.pos) > 0
    if not in_market.any():
        return float('nan'), 0

    starts = np.flatnonzero(in_market & ~np.concatenate(([False], in_market[:-1])))
    ends = np.flatnonzero(in_market & ~np.concatenate((in_market[1:], [False])))
    trade_returns: list[float] = []
    for start, end in zip(starts, ends, strict=True):
        compounded = 1.0
        for bar_return in result.net[start:end + 1]:
            compounded *= 1.0 + float(bar_return) * notional_rate
        trade_returns.append(compounded - 1.0)
    mean_bps = round(float(np.mean(trade_returns)) * BPS_PER_UNIT, BPS_DECIMALS)
    return mean_bps, len(trade_returns)


class RuleBasedStrategy(ReferenceModel):

    '''Rule-based strategy that applies boolean predicate logic per bar to produce long/flat positions.'''

    deterministic = True

    def __init__(self,
                 sharpe_std_threshold: float = 0.5,
                 sharpe_degradation_threshold: float = 0.3) -> None:

        super().__init__()
        self.sharpe_std_threshold = sharpe_std_threshold
        self.sharpe_degradation_threshold = sharpe_degradation_threshold

    @override
    def train(self, data: dict[str, Any], **params: Any) -> 'RuleBasedStrategy':

        '''
        No-op training step — rule-based strategies have no learnable parameters.

        Args:
            data (dict): Ignored
            **params: Ignored

        Returns:
            RuleBasedStrategy: Self
        '''

        return self

    @override
    def predict(self, data: dict[str, Any]) -> dict[str, Any]:

        '''
        Apply boolean logic tree to test split and return per-bar position signals.

        Args:
            data (dict): Data dict with 'test' DataFrame and 'strategy' config

        Returns:
            dict: {'_preds': np.ndarray} of 0/1 integer positions
        '''

        pos = self._apply_logic(data['test'], data['strategy']).fill_null(False).to_numpy().astype(int)
        return {'_preds': pos}

    @override
    def evaluate(self, data: dict[str, Any], inline_metrics: bool = True) -> dict[str, Any]:

        '''
        Evaluate strategy across all splits and return rule-based metrics.

        Args:
            data (dict): Data dict with 'train', 'val', 'test' DataFrames and 'strategy' config
            inline_metrics (bool): Unused — included for interface compatibility

        Returns:
            dict: Tier 1 position stats, Tier 2 backtest metrics per split,
                Tier 3 stability metrics, and '_preds' key
        '''

        from limen.experiment._backtest_provenance import preflight_backtest as _preflight_backtest

        _preflight_backtest(data)
        positions: dict[str, npt.NDArray[np.integer[Any]]] = {}
        backtest_results: dict[str, dict[str, float]] = {}
        strategy = data['strategy']
        cond_index = {c['id']: c for c in strategy['conditions']}

        cost_kwargs = self._cost_kwargs(data)
        for split in ('train', 'val', 'test'):
            pos = self._resolve(cond_index[strategy['entry']], cond_index, data[split]).fill_null(False).to_numpy().astype(int)
            positions[split] = pos
            if split == 'test' and data.get('_record_execution'):
                cost_kwargs.update(_record_execution=True, _alignment=data['_alignment'])
            backtest_results[split] = self._backtest_split(data[split], pos, cost_kwargs)

        results = rule_based_metrics(
            positions,
            backtest_results,
            sharpe_std_threshold=self.sharpe_std_threshold,
            sharpe_degradation_threshold=self.sharpe_degradation_threshold,
        )
        results['_preds'] = positions['test']
        return results

    def _apply_logic(self, df: pl.DataFrame, strategy: dict[str, Any]) -> pl.Series:
        cond_index = {c['id']: c for c in strategy['conditions']}
        return self._resolve(cond_index[strategy['entry']], cond_index, df)

    def _resolve(self,
                 condition: dict[str, Any],
                 cond_index: dict[str, dict[str, Any]],
                 df: pl.DataFrame) -> pl.Series:
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

    def _backtest_split(self,
                        df: pl.DataFrame,
                        positions: npt.NDArray[np.integer[Any]],
                        cost_kwargs: dict[str, Any]) -> dict[str, float]:
        metrics, execution_result = _evaluate_prices(df, positions, cost_kwargs)
        _record_execution(cost_kwargs, execution_result, float(cost_kwargs.get('notional_rate', 1.0)))
        if execution_result is None:
            return metrics
        pnl_per_trade_bps, executed_trade_count = _compounded_trade_pnl_summary(
            execution_result,
            float(cost_kwargs.get('notional_rate', 1.0)),
        )
        metrics['pnl_per_trade_bps'] = pnl_per_trade_bps
        metrics['num_executed_trades'] = executed_trade_count
        return metrics


def rule_based(data: dict[str, Any],
               sharpe_std_threshold: float = 0.5,
               sharpe_degradation_threshold: float = 0.3) -> dict[str, Any]:

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
