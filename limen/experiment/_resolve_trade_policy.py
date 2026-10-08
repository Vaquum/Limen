from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, cast

from limen.backtest.funding_adapter import ADAPTER_VERSION, FUNDING_KEYS, FundingAdapter, funding_adapter
from limen.backtest.funding_presets import PRESETS, PRESET_VERSION
from limen.backtest.trade_contract import FundingPolicy, JsonValue, ProductSpec, TradePolicy, finite_number, json_value
from limen.experiment._resolve_backtest_config import resolve_backtest_config


class SourceConfig(Protocol):
    @property
    def method(self) -> Callable[..., object]: ...
    @property
    def params(self) -> Mapping[str, object]: ...


@dataclass
class ProductConfig:
    kind: Literal['cash_spot', 'linear_perpetual']
    instrument: str
    base_currency: str
    quote_currency: str
    quantity_step: float | str
    min_notional: float | str
    initial_margin_fraction: float | str = 1.0
    maintenance_margin_fraction: float | str = 0.0


@dataclass
class FundingConfig:
    adapter: FundingAdapter = funding_adapter
    adapter_version: str = ADAPTER_VERSION
    preset: str | None = None
    params: dict[str, object] = field(default_factory=dict[str, object])
    data_source: SourceConfig | None = None


@dataclass
class BacktestConfig:
    fee_bps: float | str = 5.0
    slip_bps: float | str = 5.0
    notional_rate: float | str = 1.0
    take_profit_bps: float | str | None = None
    stop_loss_bps: float | str | None = None
    prediction_mode: Literal['binary', 'target_exposure'] = 'binary'
    product: ProductConfig | None = None
    initial_equity: float | str = 10000.0
    max_exposure: float | str = 1.0
    signal_change_bps: float | str = 0.0
    flat_threshold: float | str = 0.0
    max_holding_seconds: float | str | None = None
    timer_interval_seconds: float | str | None = None
    timer_phase_utc_seconds: float | str = 0.0
    execution_lag_seconds: float | str = 0.0
    max_price_gap_seconds: float | str | None = None
    execution_data_source: SourceConfig | None = None
    funding: FundingConfig | None = None


TRADE_NUMBERS = ('max_exposure', 'signal_change_bps', 'flat_threshold', 'max_holding_seconds', 'timer_interval_seconds', 'timer_phase_utc_seconds', 'execution_lag_seconds', 'max_price_gap_seconds')
BACKTEST_FIELDS = frozenset(BacktestConfig.__dataclass_fields__)


def resolve_json(value: object, params: Mapping[str, object]) -> JsonValue:
    if callable(value):
        return resolve_json(cast(Callable[[Mapping[str, object]], object], value)(params), params)
    if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
        key = value[1:-1]
        if key not in params:
            raise ValueError(f'Unknown trade search parameter {key}')
        return json_value(params[key])
    if isinstance(value, Mapping):
        return {str(key): resolve_json(item, params) for key, item in cast(Mapping[str, object], value).items()}
    if isinstance(value, (tuple, list)):
        return [resolve_json(item, params) for item in cast(list[object], value)]
    return json_value(value)


def resolve_number(value: object, params: Mapping[str, object], name: str) -> float:
    if isinstance(value, str) and value in params:
        value = params[value]
    return finite_number(resolve_json(value, params), name)


def resolve_trade_policy(config: BacktestConfig | None, params: Mapping[str, object]) -> TradePolicy | None:
    if config is None:
        return None
    enabled = config.prediction_mode != 'binary' or config.product is not None or any(value is not None for value in (config.max_holding_seconds, config.timer_interval_seconds, config.execution_data_source, config.funding, config.max_price_gap_seconds)) or any(getattr(config, name) != default for name, default in (('initial_equity', 10000.0), ('max_exposure', 1.0), ('signal_change_bps', 0.0), ('flat_threshold', 0.0), ('timer_phase_utc_seconds', 0.0), ('execution_lag_seconds', 0.0)))
    if not enabled:
        return None
    product = config.product
    if product is None:
        raise ValueError('Configured event execution requires explicit product metadata')
    product_numbers = {key: resolve_number(getattr(product, key), params, key) for key in ('quantity_step', 'min_notional', 'initial_margin_fraction', 'maintenance_margin_fraction')}
    spec = ProductSpec(product.kind, product.instrument, product.base_currency, product.quote_currency, **product_numbers)
    costs = resolve_backtest_config(config, params)
    numbers = {name: None if getattr(config, name) is None else resolve_number(getattr(config, name), params, name) for name in TRADE_NUMBERS}
    return TradePolicy(spec, prediction_mode=config.prediction_mode, fee_bps=cast(float, costs['fee_bps']), slip_bps=cast(float, costs['slip_bps']), notional_rate=cast(float, costs['notional_rate']), take_profit_bps=costs['take_profit_bps'], stop_loss_bps=costs['stop_loss_bps'], max_exposure=cast(float, numbers['max_exposure']), signal_change_bps=cast(float, numbers['signal_change_bps']), flat_threshold=cast(float, numbers['flat_threshold']), max_holding_seconds=numbers['max_holding_seconds'], timer_interval_seconds=numbers['timer_interval_seconds'], timer_phase_utc_seconds=cast(float, numbers['timer_phase_utc_seconds']), execution_lag_seconds=cast(float, numbers['execution_lag_seconds']), max_price_gap_seconds=numbers['max_price_gap_seconds'], funding=resolve_funding(config.funding, params))


def resolve_funding(config: FundingConfig | None, params: Mapping[str, object]) -> FundingPolicy | None:
    if config is None:
        return None
    if config.preset is not None and config.preset not in PRESETS:
        raise ValueError(f'Unknown funding preset {config.preset}')
    defaults = dict(PRESETS[config.preset].params) if config.preset is not None else {}
    overrides = cast(dict[str, JsonValue], resolve_json(config.params, params))
    if config.adapter is funding_adapter and set(overrides) - FUNDING_KEYS:
        raise ValueError(f'Unknown funding parameters: {set(overrides) - FUNDING_KEYS}')
    if config.data_source is not None:
        _ = defaults.pop('rate', None)
        if 'rate' in overrides:
            raise ValueError('Historical funding conflicts with a constant rate')
        if 'approximation' not in overrides:
            defaults['approximation'] = 'recorded_exact'
        if 'valuation' not in overrides:
            defaults['valuation'] = 'oracle' if config.preset == 'hyperliquid_btc' else 'mark'
    defaults.update(overrides)
    if config.preset is not None and config.data_source is None and 'rate' not in defaults:
        raise ValueError('Preset calibration publication awaits redistribution rights; provide an explicit rate or historical source')
    adapter = config.adapter
    module = getattr(adapter, '__module__', None)
    name = getattr(adapter, '__qualname__', None)
    if not isinstance(module, str) or not isinstance(name, str) or '<' in name or getattr(importlib.import_module(module), name) is not adapter:
        raise ValueError('Funding adapter must have an importable callable identity')
    mechanism, valuation, currency, approximation = (defaults.get(key) for key in ('mechanism', 'valuation', 'currency', 'approximation'))
    if mechanism not in ('discrete', 'continuous') or valuation not in ('mark', 'oracle', 'execution_proxy') or not isinstance(currency, str) or approximation not in ('scenario', 'sampled', 'recorded_exact'):
        raise ValueError('Funding requires explicit mechanism, valuation, currency and approximation')
    return FundingPolicy(f'{module}.{name}', config.adapter_version, defaults, config.preset, PRESET_VERSION if config.preset else None, mechanism, finite_number(defaults.get('rate_basis_seconds'), 'funding rate basis'), valuation, currency, approximation, PRESETS[config.preset].calibration if config.preset is not None else None)


__all__ = ['BACKTEST_FIELDS', 'TRADE_NUMBERS', 'BacktestConfig', 'FundingConfig', 'ProductConfig', 'SourceConfig', 'resolve_funding', 'resolve_json', 'resolve_number', 'resolve_trade_policy']
