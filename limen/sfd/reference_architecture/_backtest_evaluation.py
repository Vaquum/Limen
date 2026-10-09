from collections.abc import Mapping
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import polars as pl

from limen.backtest._snapshot_execution import snapshot_with_execution as _snapshot_with_execution
from limen.backtest.long_flat_strategy import ExecutionResult
from limen.backtest._long_flat_tp_sl import validate_barrier as _validate_barrier


class _ExecutionOptions(TypedDict):
    fee_bps: float
    slip_bps: float
    notional_rate: float
    take_profit_bps: float | None
    stop_loss_bps: float | None


def _cost_number(options: Mapping[str, object], key: str, default: float) -> float:
    from limen.experiment._resolve_backtest_config import validate_backtest_value as _validate_backtest_value

    value = _validate_backtest_value(key, options.get(key, default))
    if value is None:
        raise ValueError(f'backtest {key} cannot be None')
    return value


def execution_options(options: Mapping[str, object]) -> _ExecutionOptions:
    return {
        'fee_bps': _cost_number(options, 'fee_bps', 5.0),
        'slip_bps': _cost_number(options, 'slip_bps', 5.0),
        'notional_rate': _cost_number(options, 'notional_rate', 1.0),
        'take_profit_bps': _validate_barrier('take_profit_bps', options.get('take_profit_bps')),
        'stop_loss_bps': _validate_barrier('stop_loss_bps', options.get('stop_loss_bps')),
    }


def evaluate_prices(
    prices: pl.DataFrame | None, predictions: npt.ArrayLike,
    options: Mapping[str, object], *, configured: bool = False,
) -> tuple[dict[str, float], ExecutionResult | None]:
    from limen.experiment._prepare_trade_context import PreparedTradeContext, select_trade_rows
    from limen.backtest.execution_events import with_predictions
    from limen.backtest.trade_execution import trade_execution
    from limen.backtest.trade_contract import finite_number

    context = options.get('_trade_context')
    if isinstance(context, PreparedTradeContext):
        if context.policy.prediction_mode != 'binary':
            raise ValueError('Rule-based output mode does not match manifest trade policy')
        if prices is None or 'datetime' not in prices.columns:
            raise ValueError('Event evaluation requires recorded split identity')
        for inputs, mapping in zip(context.partitions, context.model_rows, strict=True):
            if prices.height and prices['datetime'].is_in(mapping['datetime'].implode()).all():
                selected = select_trade_rows(inputs, mapping, prices)
                ledger = trade_execution(with_predictions(selected, predictions), context.policy)
                completed = ledger.episodes.filter(pl.col('closed_at_ns').is_not_null())
                mean = finite_number((completed['net_pnl'] / completed['entry_equity']).mean(), 'completed episode return') * 10000 if completed.height else float('nan')
                return {**ledger.metrics, 'pnl_per_trade_bps': mean, 'num_executed_trades': float(completed.height)}, None
        raise ValueError('Event evaluation does not match a bound split')
    kwargs = execution_options(options)
    enabled = kwargs['take_profit_bps'] is not None or kwargs['stop_loss_bps'] is not None
    if prices is None or 'open' not in prices.columns or 'close' not in prices.columns:
        if enabled or configured:
            raise ValueError('price_data_for_backtest requires open/close for configured TP/SL')
        return {}, None
    columns = {col: prices[col].to_numpy() for col in ('open', 'high', 'low', 'close') if col in prices.columns}
    columns['predictions'] = np.asarray(predictions)
    columns['price_change'] = columns['close'] - columns['open']
    return _snapshot_with_execution(columns, execution_lag_bars=1, **kwargs)


def compute_backtest(predictions: npt.ArrayLike, data: Mapping[str, object]) -> dict[str, float]:
    from limen.experiment._backtest_provenance import preflight_backtest as _preflight_backtest
    from limen.experiment._resolve_backtest_config import BACKTEST_KEYS
    from limen.backtest.trade_contract import TradeInputs, TradePolicy
    from limen.backtest.execution_events import with_predictions
    from limen.backtest.trade_execution import trade_execution
    from limen.experiment._prepare_trade_context import persist_ledger

    policy, inputs = data.get('_trade_policy'), data.get('_trade_inputs')
    if policy is not None:
        if not isinstance(policy, TradePolicy) or not isinstance(inputs, TradeInputs):
            raise ValueError('Configured trade execution requires its original bound inputs')
        ledger = trade_execution(with_predictions(inputs, predictions), policy)
        persist_ledger(data, ledger)
        return {f'backtest_{key}': value for key, value in ledger.metrics.items()}

    options = {key: data[f'backtest_{key}'] for key in BACKTEST_KEYS if f'backtest_{key}' in data}
    price = data.get('price_data_for_backtest')
    if price is not None and not isinstance(price, pl.DataFrame):
        raise ValueError('price_data_for_backtest must be a DataFrame')
    _preflight_backtest(data)
    metrics, execution = evaluate_prices(price, predictions, options, configured=bool(data.get('_backtest_configured')))
    if data.get('_record_execution'):
        record_execution(data, execution, execution_options(options)['notional_rate'])
    return {f'backtest_{key}': value for key, value in metrics.items()}


def record_execution(data: Mapping[str, object], execution: ExecutionResult | None, notional_rate: float) -> None:
    if not data.get('_record_execution'):
        return
    alignment = data.get('_alignment')
    if not isinstance(alignment, dict):
        raise ValueError('Execution recording requires test alignment')
    alignment['execution'] = (
        {field: (getattr(execution, field) * notional_rate).tolist() for field in ('pos', 'gross', 'net')}
        if execution is not None else None
    )


__all__ = ['compute_backtest', 'evaluate_prices', 'execution_options', 'record_execution']
