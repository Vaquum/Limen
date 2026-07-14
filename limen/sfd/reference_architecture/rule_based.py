from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from typing_extensions import override

from limen.backtest.backtest_snapshot import backtest_snapshot
from limen.backtest.long_flat_strategy import ExecutionResult
from limen.backtest.long_flat_strategy import long_flat_strategy
from limen.metrics.rule_based_metrics import rule_based_metrics
from limen.sfd.reference_architecture.base import ReferenceModel

BPS_PER_UNIT = 10_000.0
BPS_DECIMALS = 1


def _mean_compounded_trade_pnl_bps(result: ExecutionResult, notional_rate: float) -> float:
    in_market = np.asarray(result.pos) > 0
    if not in_market.any():
        return float('nan')

    starts = np.flatnonzero(in_market & ~np.concatenate(([False], in_market[:-1])))
    ends = np.flatnonzero(in_market & ~np.concatenate((in_market[1:], [False])))
    trade_returns: list[float] = []
    for start, end in zip(starts, ends, strict=True):
        compounded = 1.0
        for bar_return in result.net[start:end + 1]:
            compounded *= 1.0 + float(bar_return) * notional_rate
        trade_returns.append(compounded - 1.0)
    return round(float(np.mean(trade_returns)) * BPS_PER_UNIT, BPS_DECIMALS)


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

        positions: dict[str, npt.NDArray[np.integer[Any]]] = {}
        backtest_results: dict[str, dict[str, float]] = {}
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
        if 'open' not in df.columns or 'close' not in df.columns:
            return {}
        open_arr = df['open'].to_numpy().astype(float)
        close_arr = df['close'].to_numpy().astype(float)
        bt_columns = {
            'predictions': positions,
            'open': open_arr,
            'close': close_arr,
            'price_change': close_arr - open_arr,
        }

        execution_result: ExecutionResult | None = None

        def capture_execution_result(predictions: Any,
                                     open_px: Any,
                                     close_px: Any,
                                     price_change: Any,
                                     *,
                                     execution_lag_bars: int,
                                     fee_bps: float,
                                     slip_bps: float) -> ExecutionResult:
            nonlocal execution_result
            execution_result = long_flat_strategy(
                predictions,
                open_px,
                close_px,
                price_change,
                execution_lag_bars=execution_lag_bars,
                fee_bps=fee_bps,
                slip_bps=slip_bps,
            )
            return execution_result

        metrics = backtest_snapshot(
            bt_columns,
            strategy=capture_execution_result,
            execution_lag_bars=1,
            **cost_kwargs,
        )
        if execution_result is None:
            raise RuntimeError('backtest strategy did not return an execution result')
        metrics['pnl_per_trade_bps'] = _mean_compounded_trade_pnl_bps(
            execution_result,
            float(cost_kwargs.get('notional_rate', 1.0)),
        )
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
