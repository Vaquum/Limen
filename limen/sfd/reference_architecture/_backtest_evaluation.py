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

    options = {key: data[f'backtest_{key}'] for key in BACKTEST_KEYS if f'backtest_{key}' in data}
    price = data.get('price_data_for_backtest')
    if price is not None and not isinstance(price, pl.DataFrame):
        raise ValueError('price_data_for_backtest must be a DataFrame')
    _preflight_backtest(data)
    metrics, _ = evaluate_prices(price, predictions, options, configured=bool(data.get('_backtest_configured')))
    return {f'backtest_{key}': value for key, value in metrics.items()}


__all__ = ['compute_backtest', 'evaluate_prices', 'execution_options']
