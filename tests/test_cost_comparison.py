from __future__ import annotations

import copy
import json
import socket
from types import MappingProxyType, SimpleNamespace
from dataclasses import replace
from decimal import Decimal
from itertools import pairwise

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from limen.backtest import _cost_comparison
from limen.backtest.funding_adapter import ADAPTER_VERSION, prepare_funding
from limen.experiment import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search.grid_strategy import GridStrategy
from limen.backtest.observed_costs import CostCalibration, CostCandidate
from limen.backtest.trade_contract import FundingPolicy, ProductSpec, TradeInputs, TradePolicy, source_binding
from limen.backtest.trade_execution import trade_execution
from tests.test_observed_costs import CUTOFF, FIXTURE, _history, _no_estimation, _no_network


def _recorded_candidate(mode, hour=1):
    benchmark = pl.DataFrame(json.loads((FIXTURE / 'benchmark.json').read_text()), schema={
        'row_id': pl.String, 'phase': pl.String, 'time_ns': pl.Int64,
        'replay_available_at_ns': pl.Int64, 'price': pl.Float64,
        'reference_id': pl.String, 'observed_receipt_ns': pl.Int64})
    first = benchmark['time_ns'][0] + (hour - 1) * 3_600_000_000_000
    selected = benchmark.filter(pl.col('time_ns').is_between(first + 180_000_000_000, first + 420_000_000_000))
    observations = selected.select('row_id', pl.col('time_ns').alias('start_ns'), pl.col('time_ns').alias('end_ns'),
        pl.col('replay_available_at_ns').alias('open_available_at_ns'), pl.col('replay_available_at_ns').alias('available_at_ns'),
        *(pl.col('price').alias(name) for name in ('open', 'high', 'low', 'close')))
    # Hypothetical strategy targets/equity; every execution price is recorded.
    targets = [1.0, 1.0, 0.0, 0.0, 0.0] if mode == 'binary' else [1.0, 1.0, -0.5, -0.5, 0.0]
    signals = selected.select('row_id', pl.col('time_ns').alias('available_at_ns')).with_columns(pl.Series('target', targets))
    start, end = selected['time_ns'][0], selected['time_ns'][-1]
    binding = source_binding(observations, 'recorded_midpoints', start, end, 1, 'retrospective_assumed_exchange_label')
    inputs = TradeInputs(10_000.0, start, end, signals, observations, None, (binding,))
    policy = TradePolicy(ProductSpec('linear_perpetual', 'BTC', 'BTC', 'USDC', 0.00001, 1.0),
                         prediction_mode=mode, fee_bps=5.0, slip_bps=5.0)
    conditions = {'venue': 'hyperliquid', 'environment': 'mainnet', 'product_kind': 'linear_perpetual',
                  'instrument': 'BTC', 'base_currency': 'BTC', 'quote_currency': 'USDC',
                  'reference_id': benchmark['reference_id'][0], 'reference_convention': 'minute_midpoint_label',
                  'clock': 'assumed_exchange_label'}
    return inputs, policy, conditions, benchmark


@pytest.fixture(scope='module')
def calibration(tmp_path_factory):
    history_path, _ = _history(tmp_path_factory.mktemp('recorded_cost_calibration'))
    return CostCalibration.from_history(history_path, cutoff_ns=CUTOFF)


@pytest.mark.parametrize('mode', ['binary', 'target_exposure'])
@pytest.mark.parametrize('funded', [False, True], ids=['unfunded', 'software-scenario-funding'])
@pytest.mark.parametrize('barrier', [None, 'take_profit_bps', 'stop_loss_bps'], ids=['no-barrier', 'software-tp', 'software-sl'])
@pytest.mark.parametrize('lag', [0.0, 60.0], ids=['immediate', 'next-recorded-minute'])
def test_frozen_event_scenarios(tmp_path, monkeypatch, mode, funded, barrier, lag, calibration):
    monkeypatch.setattr(socket, 'socket', _no_network)
    inputs, policy, conditions, benchmark = _recorded_candidate(mode)
    policy = replace(policy, execution_lag_seconds=lag, **({barrier: 0.01} if barrier else {}))
    if funded:
        funding = FundingPolicy('limen.backtest.funding_adapter.funding_adapter', ADAPTER_VERSION,
            MappingProxyType({'rate': 0.001, 'rate_unit': 'decimal', 'settlement_interval_seconds': 60.0}),
            None, None, 'discrete', 3600.0, 'execution_proxy', 'USDC', 'scenario')
        events = prepare_funding(funding, None, inputs.partition_start_ns, inputs.partition_end_ns)
        inputs = replace(inputs, funding_events=events, sources=(*inputs.sources,
            source_binding(events, 'software_scenario_funding', inputs.partition_start_ns, inputs.partition_end_ns, 1, 'hypothetical_rate_on_recorded_clock')))
        policy = replace(policy, funding=funding)
    second_policy = replace(policy, slip_bps=7.0)
    first = CostCandidate(inputs, policy, 'recorded-test:first', conditions, benchmark)
    second = CostCandidate(inputs, second_policy, 'recorded-test:second', conditions, benchmark)
    calls = []

    def execute(actual_inputs, actual_policy):
        calls.append(actual_policy)
        return trade_execution(actual_inputs, actual_policy)

    monkeypatch.setattr(_cost_comparison, 'trade_execution', execute)
    comparison = calibration.compare({'first': first, 'second': second})
    assert len(calls) == 4
    report = comparison.report()
    assert list(report['candidates']) == ['first', 'second']
    assert report['ranks']['original'] is None
    assert report['ranks']['calibrated'] == {'first': 1, 'second': 1}
    assert report['ranks']['delta'] is None
    for name, original_policy in [('first', policy), ('second', second_policy)]:
        original = comparison.ledger(name, 'original')
        calibrated = comparison.ledger(name, 'calibrated')
        expected_policy = replace(original_policy, slip_bps=float(Decimal(calibration.report()['estimates']['slip_bps'])))
        for actual, expected in [(original, trade_execution(inputs, original_policy)), (calibrated, trade_execution(inputs, expected_policy))]:
            assert actual.fills.height > 0 and actual.metrics['completed_trades'] > 0
            assert actual.metrics == expected.metrics
            for field in ('states', 'intents', 'fills', 'episodes', 'funding'):
                assert_frame_equal(getattr(actual, field), getattr(expected, field))
        assert original.metrics != calibrated.metrics
        scenario = report['candidates'][name]['scenarios']['calibrated']
        assert scenario['contract']['trade_contract']['policy']['fee_bps'] == policy.fee_bps
        assert scenario['contract']['execution_cost']['application_digest'] == report['candidates'][name]['application_digest']
        assert 'time_extrapolation' in scenario['support']['flags']
        assert scenario['disposition'] == 'retrospective_low_effective_support'
    artifact = tmp_path / 'comparison.json'
    assert comparison.write(artifact) == comparison.write(artifact) == comparison.digest
    if mode == 'binary' and not funded and barrier is None and lag == 0.0:
        _assert_refusals(calibration, monkeypatch, calls)
    # Caller aliases are discarded; public projections cannot mutate snapshots.
    inputs.signals.replace_column(inputs.signals.get_column_index('target'), pl.Series('target', [0.0] * inputs.signals.height))
    conditions['venue'] = 'metadata-fault'
    unchanged = calibration.compare({'first': first}).report()
    assert unchanged['candidates']['first'] == report['candidates']['first']
    changed_projection = comparison.ledger('first', 'calibrated')
    changed_projection.states.replace_column(changed_projection.states.get_column_index('equity'), pl.Series('equity', [0.0] * changed_projection.states.height))
    assert comparison.ledger('first', 'calibrated').states['equity'].min() > 0
    damaged = copy.deepcopy(first._snapshot)
    damaged['research_id'] = 'fault:changed-internal'
    object.__setattr__(first, '_snapshot', damaged)
    with pytest.raises(ValueError, match='Frozen candidate'):
        calibration.compare({'first': first})


class _CallerModel:
    def __init__(self):
        self.fit_count = 0

    def fit(self, frame):
        self.fit_count += 1
        self.mean = frame['price'].mean()
        return self

    def predict(self, frame, mode, scale):
        values = [1.0 if price >= self.mean else (0.0 if mode == 'binary' else -scale) for price in frame['price']]
        values[-1] = 0.0  # Explicit hypothetical terminal strategy decision.
        return pl.Series('target', values, dtype=pl.Float64)


@pytest.mark.parametrize('mode', ['binary', 'target_exposure'])
def test_python_calibration_reuse(tmp_path, monkeypatch, mode, calibration):
    monkeypatch.setattr(socket, 'socket', _no_network)
    artifact = tmp_path / 'calibration.json'
    pinned = calibration.write(artifact)
    pin_file = tmp_path / 'caller-research-specification.json'
    pin_file.write_text(json.dumps({'calibration_digest': pinned}))
    retained, comparisons = [], []
    for hour in (1, 2):
        expected = json.loads(pin_file.read_text())['calibration_digest']
        with monkeypatch.context() as context:
            from limen.backtest import _cost_statistics
            context.setattr(_cost_statistics, 'report', _no_estimation)
            loaded = CostCalibration.read(artifact, expected_digest=expected)
        inputs, policy, conditions, benchmark = _recorded_candidate(mode, hour)
        first = inputs.partition_start_ns - 180_000_000_000
        recorded = benchmark.filter(pl.col('time_ns').is_between(first, inputs.partition_end_ns))
        space = {'scale': [0.5]}

        def prep(frame, round_params):
            return {'train': frame.head(3), 'evaluation': frame.tail(5), 'scale': round_params['scale']}

        def model(data, round_params, inputs=inputs, policy=policy, hour=hour,
                  conditions=conditions, benchmark=benchmark, loaded=loaded):
            fitted = _CallerModel().fit(data['train'])
            predictions = fitted.predict(data['evaluation'], mode, round_params['scale'])
            original_bytes = predictions.to_numpy().tobytes()
            frozen = replace(inputs, signals=inputs.signals.with_columns(predictions))
            candidate = CostCandidate(frozen, policy, f'caller-owned:experiment-{hour}', conditions, benchmark)
            with monkeypatch.context() as context:
                context.setattr(fitted, 'fit', _no_estimation)
                result = loaded.compare({'caller': candidate})
            assert predictions.to_numpy().tobytes() == original_bytes
            assert fitted.fit_count == 1
            retained.append((fitted, original_bytes))
            comparisons.append(result)
            result.write(tmp_path / f'comparison-{hour}.json')
            return {'original_return': result.ledger('caller', 'original').metrics['total_return'],
                    'calibrated_return': result.ledger('caller', 'calibrated').metrics['total_return']}

        sfd = SimpleNamespace(params=lambda space=space: space, prep=prep, model=model)
        loop = UniversalExperimentLoop(data=recorded, sfd=sfd, search_strategy=GridStrategy(ParamDomain(space)))
        loop.run(str(tmp_path / f'python-experiment-{hour}'), n_permutations=1,
                 workers=1, post_processing=False, progress_bar=False)
        assert loop.manifest is None
        assert loop.experiment_log.height == 1
        assert loop.experiment_log['calibrated_return'][0] == comparisons[-1].ledger('caller', 'calibrated').metrics['total_return']
        assert comparisons[-1].ledger('caller', 'calibrated').fills.height > 0
        assert comparisons[-1].report()['calibration_digest'] == pinned
    assert len(retained) == len({id(model) for model, _ in retained}) == 2
    assert all(model.fit_count == 1 for model, _ in retained)
    first_result, later_result = [value.report()['candidates']['caller'] for value in comparisons]
    assert first_result['input_digest'] != later_result['input_digest']
    assert first_result['application_digest'] != later_result['application_digest']


def _assert_refusals(calibration, monkeypatch, calls):
    inputs, policy, conditions, benchmark = _recorded_candidate('binary')

    def candidate(data=inputs, economics=policy, declared=conditions, prices=benchmark):
        return CostCandidate(data, economics, 'software-controls:refusal-proof', declared, prices)

    shared = candidate()
    tied = calibration.compare({'a': shared, 'b': shared}).report()
    assert tied['ranks'] == {'original': {'a': 1, 'b': 1}, 'calibrated': {'a': 1, 'b': 1}, 'delta': {'a': 0, 'b': 0}}
    mixed = calibration.compare({'good': shared, 'wrong_venue': candidate(declared={**conditions, 'venue': 'metadata-fault'})}).report()
    assert mixed['ranks']['original'] == {'good': 1, 'wrong_venue': 1}
    assert mixed['ranks']['calibrated'] is mixed['ranks']['delta'] is None
    assert mixed['unavailable']['wrong_venue']['calibrated'] == ['incompatible_condition:venue']
    heterogeneous = calibration.compare({'a': shared, 'b': candidate(economics=replace(policy, fee_bps=7.0))}).report()
    assert heterogeneous['ranks'] == {'original': None, 'calibrated': None, 'delta': None}
    deleted = benchmark.filter(pl.col('row_id') != inputs.observations['row_id'][1])
    unavailable = calibration.compare({'missing_phase': candidate(prices=deleted)})
    assert unavailable.report()['candidates']['missing_phase']['scenarios']['calibrated']['ledger'] is None
    with pytest.raises(ValueError, match='no executed ledger'):
        unavailable.ledger('missing_phase', 'calibrated')
    flat = replace(inputs, signals=inputs.signals.with_columns(pl.lit(0.0).alias('target')))
    no_fills = calibration.compare({'flat': candidate(data=flat)}).report()
    assert no_fills['unavailable']['flat']['calibrated'] == ['no_executed_fills']
    assert no_fills['candidates']['flat']['scenarios']['calibrated']['metrics']['total_return'] == 0.0

    # Exact recorded point values; only the later open clock is falsified.
    points = benchmark.head(6)
    observations = points.select('row_id', pl.col('time_ns').alias('start_ns'), pl.col('time_ns').alias('end_ns'),
        pl.col('time_ns').alias('open_available_at_ns'), pl.col('time_ns').alias('available_at_ns'),
        *(pl.col('price').alias(name) for name in ('open', 'high', 'low', 'close')))
    start, end = points['time_ns'][0], points['time_ns'][-1]
    observations = observations.with_columns(
        pl.when(pl.col('end_ns') > start).then(pl.col('end_ns') - 30_000_000_000).otherwise(pl.col('end_ns')).alias('start_ns')
    ).with_columns(pl.col('start_ns').alias('open_available_at_ns'))
    signals = pl.DataFrame({'row_id': ['fault:signal0', 'fault:signal1'],
        'available_at_ns': [start + 30_000_000_000, start + 150_000_000_000], 'target': [1.0, 0.0]})
    bound = source_binding(observations, 'early_open_metadata_fault', start, end, 1, 'unchanged_midpoints_wrong_open_clock')
    early = TradeInputs(10_000.0, start, end, signals, observations, None, (bound,))
    clock_fault = calibration.compare({'early': candidate(data=early)}).report()
    assert any('unproved_observation_phase' in reason and reason.endswith(':open')
               for reason in clock_fault['unavailable']['early']['calibrated'])

    # Software endpoint-bar encoding uses two recorded points, never real extrema.
    selected = benchmark.filter(pl.col('time_ns').is_between(inputs.partition_start_ns, inputs.partition_end_ns)).to_dicts()
    bars, phases = [], []
    for index, (left, right) in enumerate(pairwise(selected)):
        name = f'endpoint-bar:{index}'
        bars.append({'row_id': name, 'start_ns': left['time_ns'], 'end_ns': right['time_ns'],
            'open_available_at_ns': left['time_ns'], 'available_at_ns': right['time_ns'],
            'open': left['price'], 'close': right['price'], 'high': max(left['price'], right['price']), 'low': min(left['price'], right['price'])})
        phases.extend({**row, 'row_id': name, 'phase': phase} for row, phase in [(left, 'open'), (right, 'close')])
    observations = pl.DataFrame(bars, schema=inputs.observations.schema)
    bound = source_binding(observations, 'software_endpoint_bars', inputs.partition_start_ns, inputs.partition_end_ns, 1, 'endpoint_encoding_not_independent_extrema')
    bar_inputs = replace(inputs, observations=observations, sources=(bound,))
    phase_frame = pl.DataFrame(phases, schema=benchmark.schema)
    positive = calibration.compare({'bars': candidate(data=bar_inputs, prices=phase_frame)}).report()
    assert positive['candidates']['bars']['scenarios']['calibrated']['disposition'] == 'retrospective_low_effective_support'
    rejected = calibration.compare({'barrier': candidate(data=bar_inputs, prices=phase_frame, economics=replace(policy, take_profit_bps=1.0))}).report()
    assert 'barrier_extrema_evidence_unavailable' in rejected['unavailable']['barrier']['calibrated']

    # Hypothetical fee stress on unchanged recorded prices; validate the whole family first.
    expensive = replace(policy, fee_bps=20_000.0)
    invalid = replace(inputs, signals=inputs.signals.with_columns(pl.Series('target', [1.0, 0.0, 2.0, 0.0, 0.0])))
    before = len(calls)
    with pytest.raises(ValueError, match='exposure mode/bounds'):
        calibration.compare({'good': shared, 'malformed_tail': candidate(data=invalid, economics=expensive)})
    assert len(calls) == before
    valid = replace(invalid, signals=invalid.signals.with_columns(pl.Series('target', [1.0, 0.0, 0.0, 0.0, 0.0])))
    failed = calibration.compare({'economic': candidate(data=valid, economics=expensive)}).report()
    for scenario in ('original', 'calibrated'):
        result = failed['candidates']['economic']['scenarios'][scenario]
        assert result['disposition'] == 'economic_failure' and result['metrics'] is None
        assert result['support']['failure_time_ns'] is None
        assert failed['failures']['economic'][scenario]

    funding = FundingPolicy('limen.backtest.funding_adapter.funding_adapter', ADAPTER_VERSION,
        {'rate': 0.001, 'rate_unit': 'decimal', 'settlement_interval_seconds': 60.0},
        None, None, 'discrete', 3600.0, 'execution_proxy', 'USDC', 'scenario')
    events = prepare_funding(funding, None, inputs.partition_start_ns, inputs.partition_end_ns).with_columns(
        pl.when(pl.col('time_ns') == inputs.partition_end_ns).then(0.0).otherwise(pl.col('rate_basis_seconds')).alias('rate_basis_seconds'))
    bound = source_binding(events, 'software_funding_basis_fault', inputs.partition_start_ns, inputs.partition_end_ns, 1, 'invalid_normalized_basis')
    funded = replace(valid, funding_events=events, sources=(*valid.sources, bound))
    before = len(calls)
    with pytest.raises(ValueError, match='Funding basis must be positive'):
        calibration.compare({'bad_funding_tail': candidate(data=funded, economics=replace(expensive, funding=funding))})
    distances = valid.signals.with_columns(pl.Series('tp_distance', [1.0, 1.0, 1.0, 1.0, 0.0]), pl.lit(True).alias('__barrier_ready__'))
    with pytest.raises(ValueError, match='Malformed ready barrier distance'):
        calibration.compare({'bad_barrier_tail': candidate(data=replace(valid, signals=distances), economics=replace(expensive, take_profit_bps_column='tp_distance'))})
    assert len(calls) == before

    signed, signed_policy, declared, prices = _recorded_candidate('target_exposure')
    # Larger hypothetical capital avoids the venue minimum masking this cost stress.
    signed = replace(signed, initial_equity=1_000_000.0, signals=signed.signals.with_columns(pl.Series('target', [-1.0, 0.0, 0.0, 0.0, 0.0])))
    recovery = calibration.compare({name: CostCandidate(signed, economics, 'software-controls:short', declared, prices)
        for name, economics in [('wide', replace(signed_policy, slip_bps=9999.0)), ('normal', signed_policy)]}).report()
    assert recovery['candidates']['wide']['scenarios']['original']['disposition'] == 'economic_failure'
    assert recovery['ranks']['original'] is recovery['ranks']['delta'] is None
    assert recovery['ranks']['calibrated'] == {'wide': 1, 'normal': 1}

    def unexpected(*args, **kwargs):
        raise ValueError('unexpected kernel defect')

    with monkeypatch.context() as context:
        context.setattr(_cost_comparison, 'trade_execution', unexpected)
        with pytest.raises(ValueError, match='unexpected kernel defect'):
            calibration.compare({'valid': shared})
    with pytest.raises(TypeError, match='mapping'):
        calibration.compare([])
    with pytest.raises(ValueError, match='at least one'):
        calibration.compare({})
