from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
from typing import Literal, TypeAlias, cast

import polars as pl

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list['JsonValue'] | dict[str, 'JsonValue']
PredictionMode: TypeAlias = Literal['binary', 'target_exposure']
RULE_VERSION = 'limen-trade-v1'
BPS = 10000.0
NANOSECONDS = 1000000000


def finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


@dataclass(frozen=True)
class ProductSpec:
    kind: Literal['cash_spot', 'linear_perpetual']
    instrument: str
    base_currency: str
    quote_currency: str
    quantity_step: float
    min_notional: float
    initial_margin_fraction: float = 1.0
    maintenance_margin_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ('cash_spot', 'linear_perpetual'):
            raise ValueError('Unsupported product accounting')
        if not all((self.instrument, self.base_currency, self.quote_currency)):
            raise ValueError('Product instrument and currencies are required')
        for name in ('quantity_step', 'min_notional', 'initial_margin_fraction', 'maintenance_margin_fraction'):
            _ = finite_number(getattr(self, name), name)
        if self.quantity_step <= 0 or self.min_notional < 0:
            raise ValueError('Quantity step must be positive and minimum notional nonnegative')
        if not 0 < self.initial_margin_fraction <= 1 or not 0 <= self.maintenance_margin_fraction < self.initial_margin_fraction:
            raise ValueError('Invalid initial/maintenance margin fractions')


@dataclass(frozen=True)
class SourceBinding:
    identity: str
    checksum: str
    start_ns: int
    end_ns: int
    precision_ns: int
    interpretation: str


@dataclass(frozen=True)
class FundingPolicy:
    adapter_ref: str
    adapter_version: str
    params: Mapping[str, JsonValue]
    preset_id: str | None
    preset_version: str | None
    mechanism: Literal['discrete', 'continuous']
    rate_basis_seconds: float
    valuation: Literal['mark', 'oracle', 'execution_proxy']
    currency: str
    approximation: Literal['scenario', 'sampled', 'recorded_exact']
    calibration: Mapping[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        if not self.adapter_ref or not self.adapter_version or '<locals>' in self.adapter_ref:
            raise ValueError('Funding adapter export requires importable identity and version')
        if self.mechanism not in ('discrete', 'continuous') or self.valuation not in ('mark', 'oracle', 'execution_proxy'):
            raise ValueError('Invalid funding mechanism or valuation')
        if self.approximation not in ('scenario', 'sampled', 'recorded_exact') or not self.currency:
            raise ValueError('Funding approximation and currency are required')
        if finite_number(self.rate_basis_seconds, 'funding rate basis') <= 0:
            raise ValueError('Funding rate basis must be positive')
        if self.approximation == 'recorded_exact' and self.valuation == 'execution_proxy':
            raise ValueError('Execution-price proxy cannot establish exact funding')


@dataclass(frozen=True)
class TradePolicy:
    product: ProductSpec
    prediction_mode: PredictionMode = 'target_exposure'
    rule_version: str = RULE_VERSION
    fee_bps: float = 5.0
    slip_bps: float = 5.0
    notional_rate: float = 1.0
    max_exposure: float = 1.0
    signal_change_bps: float = 0.0
    flat_threshold: float = 0.0
    take_profit_bps: float | None = None
    stop_loss_bps: float | None = None
    max_holding_seconds: float | None = None
    timer_interval_seconds: float | None = None
    timer_phase_utc_seconds: float = 0.0
    execution_lag_seconds: float = 0.0
    max_price_gap_seconds: float | None = None
    funding: FundingPolicy | None = None

    def __post_init__(self) -> None:
        if self.rule_version != RULE_VERSION or self.prediction_mode not in ('binary', 'target_exposure'):
            raise ValueError('Unsupported trade rule version or prediction mode')
        for name in ('fee_bps', 'slip_bps', 'signal_change_bps', 'flat_threshold', 'execution_lag_seconds'):
            if finite_number(getattr(self, name), name) < 0:
                raise ValueError(f'{name} must be nonnegative')
        if self.slip_bps >= BPS or self.flat_threshold >= 1 or self.signal_change_bps >= BPS:
            raise ValueError('Invalid slippage/flat/signal-change bounds')
        for name in ('notional_rate', 'max_exposure'):
            if not 0 < finite_number(getattr(self, name), name) <= 1:
                raise ValueError(f'{name} must be in (0, 1]')
        for name in ('max_holding_seconds', 'timer_interval_seconds', 'max_price_gap_seconds', 'take_profit_bps', 'stop_loss_bps'):
            value = getattr(self, name)
            if value is not None and finite_number(value, name) <= 0:
                raise ValueError(f'{name} must be positive or disabled')
        if any(value is not None and value >= BPS for value in (self.take_profit_bps, self.stop_loss_bps)):
            raise ValueError('Signed price barriers must be below 10000 bps')
        _ = finite_number(self.timer_phase_utc_seconds, 'timer phase')
        if self.funding is not None and (self.product.kind != 'linear_perpetual' or self.funding.currency != self.product.quote_currency):
            raise ValueError('Funding requires a linear perpetual in the declared quote currency')


@dataclass(frozen=True)
class TradeInputs:
    initial_equity: float
    partition_start_ns: int
    partition_end_ns: int
    signals: pl.DataFrame
    observations: pl.DataFrame
    funding_events: pl.DataFrame | None
    sources: tuple[SourceBinding, ...]


@dataclass(frozen=True)
class ExecutionEvent:
    event_id: str
    time_ns: int
    kind: Literal['observation', 'funding', 'timer', 'signal']
    source_row_id: str | None
    available_at_ns: int
    observation_phase: Literal['open', 'close'] | None = None


@dataclass(frozen=True)
class TradeLedger:
    states: pl.DataFrame
    intents: pl.DataFrame
    fills: pl.DataFrame
    episodes: pl.DataFrame
    funding: pl.DataFrame
    metrics: Mapping[str, float]
    contract_digest: str


def json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Nonfinite contract value')
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if any(not isinstance(key, str) for key in mapping):
            raise ValueError('Contract mapping keys must be strings')
        return {str(key): json_value(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in cast(list[object] | tuple[object, ...], value)]
    raise ValueError(f'Nonserializable contract value: {type(value).__name__}')


def contract_digest(contract: Mapping[str, JsonValue]) -> str:
    return sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_binding(frame: pl.DataFrame, identity: str, start_ns: int, end_ns: int, precision_ns: int, interpretation: str) -> SourceBinding:
    payload = str(frame.schema).encode() + frame.hash_rows(seed=0).to_numpy().tobytes()
    return SourceBinding(identity, sha256(payload).hexdigest(), start_ns, end_ns, precision_ns, interpretation)


def export_trade_contract(policy: TradePolicy, inputs: TradeInputs) -> dict[str, JsonValue]:
    return {
        'rule_version': RULE_VERSION,
        'policy': json_value(policy),
        'calibration': json_value(policy.funding.calibration) if policy.funding is not None else None,
        'initial_conditions': {'equity': inputs.initial_equity, 'quantity': 0.0, 'currency': policy.product.quote_currency},
        'sources': json_value(inputs.sources),
        'signal_rows': json_value(inputs.signals.select('row_id', 'available_at_ns').to_dicts()),
        'partition': [inputs.partition_start_ns, inputs.partition_end_ns],
        'rules': {
            'sizing': 'signal_change_post_cost_equity',
            'rounding': 'toward_zero',
            'event_order': 'funding_barrier_sl_first_timeout_signal_exit_resize',
            'timer_origin': 'utc_epoch',
            'forced_exit_reentry': 'explicit_flat',
            'episode_origin': 'first_fill_until_full_closure',
            'terminal': 'mark_without_close',
            'ohlc': 'adverse_gap_sl_first_interval_uncertainty',
        },
    }


__all__ = ['BPS', 'NANOSECONDS', 'RULE_VERSION', 'ExecutionEvent', 'FundingPolicy', 'JsonScalar', 'JsonValue', 'PredictionMode', 'ProductSpec', 'SourceBinding', 'TradeInputs', 'TradeLedger', 'TradePolicy', 'contract_digest', 'export_trade_contract', 'finite_number', 'json_value', 'source_binding']
