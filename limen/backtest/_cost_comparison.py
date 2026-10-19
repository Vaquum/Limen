from __future__ import annotations

import copy
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from importlib import import_module
from typing import cast

import polars as pl

from limen.backtest import _cost_history as history
from limen.backtest import _cost_identity as identity
from limen.backtest import _cost_statistics as statistics
from limen.backtest.execution_events import source_execution_events, validate_signal_barriers
from limen.backtest.funding_adapter import FUNDING_KEYS, funding_adapter, resolve_adapter
from limen.backtest.trade_contract import JsonValue, TradeInputs, TradeLedger, TradePolicy, export_trade_contract, finite_number, json_value
from limen.backtest.trade_execution import trade_execution

_SCHEMA = {'row_id': pl.String, 'phase': pl.String, 'time_ns': pl.Int64,
           'replay_available_at_ns': pl.Int64, 'price': pl.Float64,
           'reference_id': pl.String, 'observed_receipt_ns': pl.Int64}
_CONDITIONS = frozenset(('venue', 'environment', 'product_kind', 'instrument', 'base_currency',
                         'quote_currency', 'reference_id', 'reference_convention', 'clock'))
_ECONOMIC = ('No affordable post-cost target quantity with a positive solution denominator',
             'Fill is unaffordable under collateral/cost rules')
_canonical_signal = cast(Callable[[object, TradePolicy], float | None],
                         vars(import_module('limen.backtest.trade_execution'))['_canonical_signal'])


def freeze(inputs: object, policy: object, research_id: str,
           conditions: object, benchmark: object) -> dict[str, object]:
    if not isinstance(inputs, TradeInputs) or not isinstance(policy, TradePolicy) or not isinstance(benchmark, pl.DataFrame):
        raise TypeError('CostCandidate requires TradeInputs, TradePolicy and a benchmark DataFrame')
    name = history.text(research_id, 'research_id')
    if ':' not in name:
        raise ValueError('research_id must be namespaced')
    declared = history.mapping(conditions, _CONDITIONS)
    for key, value in declared.items():
        _ = history.text(value, key)
    if declared['clock'] != 'assumed_exchange_label':
        raise ValueError('Only the explicit retrospective assumed-exchange clock is available')
    snapshot: dict[str, object] = copy.deepcopy({'inputs': inputs, 'policy': policy, 'research_id': name,
                              'conditions': declared, 'benchmark': benchmark})
    _ = identity.digest(snapshot)
    return snapshot


def _preflight(snapshot: Mapping[str, object]) -> None:
    inputs, policy = cast(TradeInputs, snapshot['inputs']), cast(TradePolicy, snapshot['policy'])
    funding = replace(policy.funding) if policy.funding is not None else None
    _ = replace(policy, product=replace(policy.product), funding=funding)
    _ = source_execution_events(inputs, policy, include_timers=False)
    validate_signal_barriers(inputs, policy)
    for target in inputs.signals['target']:
        signal = _canonical_signal(target, policy)
        if signal is not None and (abs(signal * policy.notional_rate) > policy.max_exposure
                or (signal < 0 and policy.product.kind == 'cash_spot')):
            raise ValueError('Target exceeds the declared product/exposure bounds')
    for source in inputs.sources:
        _ = history.digest_id(source.checksum)
        _ = history.text(source.identity, 'source identity')
        _ = history.text(source.interpretation, 'source interpretation')
        if history.integer(source.start_ns, 'source start') > history.integer(source.end_ns, 'source end'):
            raise ValueError('Source binding interval is reversed')
        _ = history.integer(source.precision_ns, 'source precision', minimum=1)
    if funding is not None and resolve_adapter(funding) is funding_adapter and set(funding.params) - FUNDING_KEYS:
        raise ValueError('Unknown native funding parameters')
    if inputs.funding_events is not None:
        if funding is None:
            raise ValueError('Funding events require their resolved policy')
        for row in inputs.funding_events.iter_rows(named=True):
            start, end, at = (history.integer(row[key], key) for key in ('start_ns', 'end_ns', 'time_ns'))
            kind = row['kind']
            if (end < start or (kind == 'accrual' and (start == end or at != start or funding.mechanism != 'continuous'))
                    or (kind != 'accrual' and (start != end or at != end))
                    or (kind == 'payment' and funding.mechanism != 'discrete')
                    or (kind == 'cash_settlement' and funding.mechanism != 'continuous')):
                raise ValueError('Funding interval/kind contradicts its resolved policy')
            _ = finite_number(row['rate_decimal'], 'funding rate')
            if finite_number(row['rate_basis_seconds'], 'funding basis') <= 0:
                raise ValueError('Funding basis must be positive')
            if row['valuation_price'] is None:
                if kind != 'cash_settlement' and (funding.valuation != 'execution_proxy' or funding.approximation == 'recorded_exact'):
                    raise ValueError('Funding row requires its recorded valuation')
            elif finite_number(row['valuation_price'], 'funding valuation') <= 0:
                raise ValueError('Funding valuation must be positive')


def _benchmark(snapshot: Mapping[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    frame = cast(pl.DataFrame, snapshot['benchmark'])
    if dict(frame.schema) != _SCHEMA or frame.null_count().sum_horizontal()[0]:
        raise ValueError('Benchmark requires the exact non-null version1 schema')
    reference = history.mapping(snapshot['conditions'])['reference_id']
    rows: dict[tuple[str, str], dict[str, object]] = {}
    previous: dict[str, int] = {}
    for row in frame.iter_rows(named=True):
        key = (history.text(row['row_id'], 'benchmark row'), history.text(row['phase'], 'phase'))
        at = history.integer(row['time_ns'], 'benchmark time')
        if key in rows or key[1] not in ('point', 'open', 'close') or at <= previous.get(key[1], -1):
            raise ValueError('Benchmark phases must have unique keys and increasing times')
        if (row['reference_id'] != reference or finite_number(row['price'], 'benchmark price') <= 0
                or history.integer(row['replay_available_at_ns'], 'replay time') < at
                or history.integer(row['observed_receipt_ns'], 'benchmark receipt') < at):
            raise ValueError('Benchmark reference/value/clock is inconsistent')
        rows[key], previous[key[1]] = row, at
    return rows


def _admission(snapshot: Mapping[str, object], calibration: Mapping[str, JsonValue], rows: Mapping[tuple[str, str], Mapping[str, object]]) -> list[str]:
    report = history.mapping(calibration['report'])
    source = history.mapping(report['source'])
    product = history.mapping(source['product'])
    conditions = history.mapping(snapshot['conditions'])
    inputs, policy = cast(TradeInputs, snapshot['inputs']), cast(TradePolicy, snapshot['policy'])
    expected = {**{key: source[key] for key in ('venue', 'environment', 'reference_id', 'reference_convention')},
                'product_kind': product['kind'], **{key: product[key] for key in ('instrument', 'base_currency', 'quote_currency')}}
    reasons = [f'incompatible_condition:{key}' for key, value in expected.items() if conditions[key] != value]
    for key, value in product.items():
        if getattr(policy.product, key) != value:
            reasons.append(f'incompatible_product:{key}')
    reasons.extend(cast(list[str], history.mapping(report['support'])['reasons']))
    has_barrier = any(getattr(policy, name) is not None for name in ('take_profit_bps', 'stop_loss_bps', 'take_profit_bps_column', 'stop_loss_bps_column'))
    for row in inputs.observations.iter_rows(named=True):
        point = row['start_ns'] == row['end_ns']
        phases = [('point' if point else 'close', 'end_ns', 'available_at_ns', 'close')]
        if not point and row['open_available_at_ns'] == row['start_ns']:
            phases.append(('open', 'start_ns', 'open_available_at_ns', 'open'))
        if not point and has_barrier:
            reasons.append('barrier_extrema_evidence_unavailable')
        for phase, clock, available, price in phases:
            matched = rows.get((str(row['row_id']), phase))
            if matched is None or any((matched['time_ns'] != row[clock], matched['replay_available_at_ns'] != row[available],
                                       float(cast(float, matched['price'])).hex() != float(row[price]).hex())):
                reasons.append(f'unproved_observation_phase:{row["row_id"]}:{phase}')
    return sorted(set(reasons))


def wire(ledger: TradeLedger) -> dict[str, JsonValue]:
    return {**{name: json_value(getattr(ledger, name).to_dicts()) for name in ('states', 'intents', 'fills', 'episodes', 'funding')},
            'metrics': json_value(ledger.metrics), 'contract_digest': ledger.contract_digest}


def _support(ledger: TradeLedger, report: Mapping[str, JsonValue], rows: Mapping[tuple[str, str], Mapping[str, object]]) -> dict[str, JsonValue]:
    source = history.mapping(report['source'])
    estimates = history.mapping(report['estimates'])
    flags: set[str] = set()
    intents = {row['intent_id']: row['reason'] for row in ledger.intents.iter_rows(named=True)}
    for fill in ledger.fills.iter_rows(named=True):
        side = 'buy' if fill['quantity_delta'] > 0 else 'sell'
        estimate = history.mapping(estimates[side])
        quantity = abs(fill['quantity_delta'])
        if not float(cast(str, estimate['quantity_min'])) <= quantity <= float(cast(str, estimate['quantity_max'])):
            flags.add('size_extrapolation:' + side)
        if not any(start <= fill['time_ns'] < end for start, end in cast(list[list[int]], source['windows'])):
            flags.add('time_extrapolation')
        barrier = intents[fill['intent_id']] in ('take_profit', 'stop_loss')
        matched = [row for (name, _), row in rows.items() if name == fill['evidence_row_id']
                   and row['replay_available_at_ns'] == fill['time_ns']
                   and (barrier or float(cast(float, row['price'])).hex() == float(fill['reference_price']).hex())]
        if not matched:
            flags.add('unproved_executed_reference')
        if barrier:
            flags.add('software_barrier_stress')
    return history.mapping({'fill_count': ledger.fills.height, 'flags': sorted(flags), 'failure_time_ns': None})


def _run(snapshot: Mapping[str, object], policy: TradePolicy, calibration_digest: str,
         report: Mapping[str, JsonValue], rows: Mapping[tuple[str, str], Mapping[str, object]], *, calibrated: bool) -> tuple[dict[str, JsonValue], TradeLedger | None]:
    inputs = cast(TradeInputs, snapshot['inputs'])
    support: dict[str, JsonValue] = {'failure_time_ns': None}
    disposition = cast(str, history.mapping(report['support'])['disposition']) if calibrated else 'original'
    reasons: list[str] = []
    ledger = None
    try:
        ledger = trade_execution(inputs, policy)
    except ValueError as error:
        if str(error) not in _ECONOMIC and not str(error).startswith('Nonpositive equity or maintenance margin breach:'):
            raise
        disposition, reasons = 'economic_failure', [str(error)]
    if ledger is not None:
        support = _support(ledger, report, rows) if calibrated else {'fill_count': ledger.fills.height}
        if calibrated and (not ledger.fills.height or 'unproved_executed_reference' in cast(list[str], support['flags'])):
            disposition, reasons = 'unavailable', ['no_executed_fills' if not ledger.fills.height else 'unproved_executed_reference']
    contract = {'trade_contract': export_trade_contract(policy, inputs), 'execution_cost': {
        'calibration_digest': calibration_digest, 'disposition': disposition, 'clock': 'assumed_exchange_label'}}
    return history.mapping({'disposition': disposition, 'reasons': reasons, 'clock': 'assumed_exchange_label',
            'metrics': json_value(ledger.metrics) if ledger is not None else None,
            'ledger': wire(ledger) if ledger is not None else None, 'contract': contract, 'support': support}), ledger


def compare(calibration: Mapping[str, JsonValue], calibration_digest: str,
            candidates: Mapping[str, Mapping[str, object]]) -> tuple[dict[str, JsonValue], dict[tuple[str, str], TradeLedger]]:
    prepared: dict[str, dict[tuple[str, str], dict[str, object]]] = {}
    for name, snapshot in candidates.items():
        _preflight(snapshot)
        prepared[name] = _benchmark(snapshot)
    report = history.mapping(calibration['report'])
    estimates = history.mapping(report['estimates'])
    payload: dict[str, JsonValue] = {'calibration_digest': calibration_digest, 'created_at_ns': time.time_ns(),
        'order': list(candidates), 'candidates': {}, 'ranks': {}, 'failures': {}, 'unavailable': {},
        'limitations': copy.deepcopy(report['limitations'])}
    ledgers: dict[tuple[str, str], TradeLedger] = {}
    results: dict[str, JsonValue] = {}
    for name, snapshot in candidates.items():
        inputs, policy = cast(TradeInputs, snapshot['inputs']), cast(TradePolicy, snapshot['policy'])
        reasons = _admission(snapshot, calibration, prepared[name])
        scenarios: dict[str, JsonValue] = {}
        for scenario in ('original', 'calibrated'):
            if scenario == 'calibrated' and reasons:
                result, ledger = history.mapping({'disposition': 'unavailable', 'reasons': reasons, 'clock': 'assumed_exchange_label',
                    'metrics': None, 'ledger': None, 'contract': None, 'support': {}}), None
            else:
                resolved = replace(policy, slip_bps=float(cast(str, estimates['slip_bps']))) if scenario == 'calibrated' else policy
                result, ledger = _run(snapshot, resolved, calibration_digest, report, prepared[name], calibrated=scenario == 'calibrated')
            scenarios[scenario] = result
            if ledger is not None:
                ledgers[name, scenario] = ledger
        item = {'research_id': snapshot['research_id'], 'conditions': snapshot['conditions'],
                'input_digest': identity.digest(inputs), 'benchmark_digest': identity.digest(snapshot['benchmark']),
                'original_policy': json_value(policy), 'scenarios': scenarios}
        application = identity.digest({'version': 1, 'calibration_digest': calibration_digest, **item})
        for label, scenario in scenarios.items():
            row = history.mapping(scenario)
            if row['contract'] is not None:
                contract = history.mapping(row['contract'])
                contract['execution_cost'] = {**history.mapping(contract['execution_cost']), 'application_digest': application}
                row['contract'] = contract
            scenarios[label] = row
        results[name] = history.mapping({'application_digest': application, **item})
    payload['candidates'] = results
    payload['ranks'] = statistics.ranks(candidates, results)
    for field, status in (('failures', 'economic_failure'), ('unavailable', 'unavailable')):
        payload[field] = {name: {scenario: history.mapping(value)['reasons'] for scenario, value in history.mapping(history.mapping(item)['scenarios']).items()
                               if history.mapping(value)['disposition'] == status} for name, item in results.items()}
    return payload, ledgers



__all__: list[str] = []
