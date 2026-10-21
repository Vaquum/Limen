import copy
import hashlib
import json
import math
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
import pytest

from limen.backtest.execution_events import with_predictions
from limen.backtest.trade_contract import BPS, TradeInputs, contract_digest
from limen.backtest.trade_execution import trade_execution
from limen.cohort import Cohort
from limen.experiment import UniversalExperimentLoop
from limen.experiment import _holdout_assessment as assessment
from limen.experiment import _holdout_events as events
from limen.experiment import _holdout_history as history
from limen.experiment import _round_storage
from limen.experiment._holdout_freeze import frozen_parts, owned_copy
from limen.experiment._holdout_source import load_admitted_source
from limen.experiment._prepare_trade_context import finish_trade_result, normalize_observations, prepare_trade_context
from limen.experiment._resolve_trade_policy import holdout_event_policy, holdout_structural
from limen.experiment.holdout import final_holdout_report, freeze_holdout_selection, read_holdout_report, reserve_holdout
from limen.inference import Trainer
from limen.targets.next_bar_up import NextBarUpTarget
from limen.yaml import CompiledSFD
from limen.yaml.compiler import build_search_strategy
from tests.test_holdout_assessment import _SCENARIO
from tests.test_holdout_research import _FIXTURE, _configuration, _project

_SPOT = {'kind': 'cash_spot', 'instrument': 'BTCUSDT', 'base_currency': 'BTC', 'quote_currency': 'USDT', 'quantity_step': 1e-9, 'min_notional': 0}
_PERP = {'kind': 'linear_perpetual', 'instrument': 'BTCUSDT', 'base_currency': 'BTC', 'quote_currency': 'USDT', 'quantity_step': 1e-9, 'min_notional': 0}
_EXECUTION_KEYS = {'research_digest', 'assessment_digest', 'policy', 'initial_equity', 'metrics', 'ledger', 'contract', 'tape', 'signals'}


def _attempts(root):
    with history.connection(root / 'manifests/holdout.sqlite3') as db:
        return db.execute('SELECT state, error_json, context_start_ns, start_ns, end_ns FROM attempts ORDER BY assessment_index').fetchall()


def _locator(case, name):
    return {'experiment_dir': str(case['directories'][name]), 'round_id': case['rounds'][name]['round_id']}


def _freeze(case, name, **overrides):
    arguments = {'context_start': '2024-01-04', 'scenarios': [], 'evaluation_target': NextBarUpTarget(pl.DataFrame(), 'observed_up'),
                 'round_locator': _locator(case, name)}
    arguments.update(overrides)
    return freeze_holdout_selection(case['sensors'][name], **arguments)


@pytest.fixture(scope='module')
def event_case(tmp_path_factory):
    root = _project(tmp_path_factory.mktemp('event') / 'project')
    base = _configuration(root)
    spot = copy.deepcopy(base)
    spot['sfd']['manifest']['backtest'] = {'product': dict(_SPOT), 'notional_rate': 0.25}
    perp = copy.deepcopy(base)
    perp['sfd']['manifest']['backtest'] = {'product': dict(_PERP), 'initial_equity': '{equity}', 'fee_bps': '{fee}'}
    perp['sfd']['params']['equity'] = [25000.0]
    perp['sfd']['params']['fee'] = [5.0]
    reservation = reserve_holdout(root, '2024-01-05', '2024-01-07', interval_seconds=3600)
    sensors, rounds, directories = {}, {}, {}
    for name, config in (('spot', spot), ('perp', perp)):
        directory = root / name
        loop = UniversalExperimentLoop(sfd=CompiledSFD(config, project_root=root), project_root=root, experiment_dir=directory,
            search_strategy=build_search_strategy(config), yaml_reference=config)
        loop.run(name, n_permutations=1, progress_bar=False, post_processing=True)
        with _round_storage.open_text(_round_storage.round_path(directory)) as stream:
            record = next(json.loads(line) for line in stream if line.strip())
        rounds[name] = record
        directories[name] = directory
        sensors[name] = Trainer(directory, project_root=root).train([record['round_id']])[0]
    case = {'root': root, 'reservation': reservation, 'sensors': sensors, 'rounds': rounds, 'directories': directories,
            'configs': {'spot': spot, 'perp': perp}}
    case['reports'] = {name: final_holdout_report(_freeze(case, name), project_root=root, reservation_id=reservation, end='2024-01-06')
                       for name in ('spot', 'perp')}
    return case


def test_recorded_event_assessment_round_trip(event_case):
    report = event_case['reports']['spot']
    assert report['schema_version'] == 2
    assert report['scenarios'] == []
    execution = report['execution']
    assert set(execution) == _EXECUTION_KEYS
    assert set(execution['ledger']) == {'states', 'intents', 'fills', 'episodes', 'funding'}
    assert execution['ledger']['funding'] == []
    assert execution['research_digest'] == event_case['rounds']['spot']['trade_contract_digest']
    assert execution['initial_equity'] == 10000.0
    assert execution['policy'] == report['spec']['members'][0]['execution']['policy']
    assert contract_digest(execution['contract']) == execution['assessment_digest']
    assert execution['contract']['partition'] == [report['planned']['context_start_ns'], report['planned']['end_ns']]
    assert len(execution['tape']) == len(execution['signals']) == 24
    assert [point['prediction'] for point in execution['tape']] == [int(signal['target']) for signal in execution['signals']]
    back = read_holdout_report(event_case['root'], report['attempt_id'])
    assert back['schema_version'] == 2
    assert back['execution'] == execution


def _replay_inputs(case, name):
    report = case['reports'][name]
    sensor = case['sensors'][name]
    manifest = sensor._get_manifest()
    config = manifest.backtest_config
    params = dict(sensor._round_params)
    policy = holdout_event_policy(config, params)
    binding = history.ProjectBinding(case['root'], report['project_id'], report['reservation']['revision'], None, 3600)
    planned = report['planned']
    raw, _ = load_admitted_source(manifest, binding, lower_ns=planned['context_start_ns'], upper_ns=planned['end_ns'], complete=True)
    final = raw.filter(pl.col('datetime').dt.epoch('ns') >= planned['start_ns'])
    frame = normalize_observations(raw, interval_seconds=3600, timestamp_role=config.source_timestamp_role)
    prepared = prepare_trade_context(config, policy, [raw], [final.select('datetime')], params, interval_seconds=3600, frozen_sources=(frame, None))
    return prepared.partitions[0], policy


def _assert_replay_matches(case, name):
    execution = case['reports'][name]['execution']
    inputs, policy = _replay_inputs(case, name)
    bound = with_predictions(replace(inputs, initial_equity=execution['initial_equity']), [point['prediction'] for point in execution['tape']])
    ledger = trade_execution(bound, policy)
    assert ledger.contract_digest == execution['assessment_digest']
    for frame in ('states', 'intents', 'fills', 'episodes', 'funding'):
        assert getattr(ledger, frame).to_dicts() == execution['ledger'][frame]
    for metric, value in ledger.metrics.items():
        field = execution['metrics'][metric]
        assert field['value'] == (float(value) if math.isfinite(value) else None)
        assert field['support_rows'] == ledger.fills.height
    return inputs, policy


def test_recorded_event_ledger_matches_engine_replay(event_case):
    spot, policy = _assert_replay_matches(event_case, 'spot')
    equity = event_case['reports']['spot']['execution']['initial_equity']
    quiet = trade_execution(with_predictions(replace(spot, initial_equity=equity), [0] * 24), policy)
    assert quiet.fills.height == 0
    assert all(value == 0 for value in quiet.states['quantity'].to_list())
    assert quiet.metrics['ending_equity'] == equity
    held = trade_execution(with_predictions(replace(spot, initial_equity=equity), [1] * 24), policy)
    episodes = held.episodes.to_dicts()
    assert any(episode['closed_at_ns'] is None for episode in episodes)
    carried = next(episode for episode in episodes if episode['closed_at_ns'] is None)
    assert carried['net_pnl'] == held.states['equity'][-1] - carried['entry_equity']
    _assert_replay_matches(event_case, 'perp')
    spot_policy = event_case['reports']['spot']['execution']['policy']
    perp_policy = event_case['reports']['perp']['execution']['policy']
    assert spot_policy['product']['kind'] == 'cash_spot'
    assert perp_policy['product']['kind'] == 'linear_perpetual'
    assert perp_policy['product']['initial_margin_fraction'] == 1.0
    ledgers = {name: event_case['reports'][name]['execution']['ledger'] for name in ('spot', 'perp')}
    assert ledgers['spot']['states'][0]['equity'] != ledgers['perp']['states'][0]['equity']
    targets = [intent['target'] for intent in ledgers['spot']['intents']]
    assert 0.25 in targets
    assert all(target in (0.0, 0.25) for target in targets)
    root, report = event_case['root'], event_case['reports']['spot']
    _rewrite_report(root, report, lambda tampered: tampered['execution'].update(ledger=copy.deepcopy(ledgers['perp'])), 'accounting anchor')

    def drift(report_dict):
        report_dict['execution']['tape'][5]['prediction'] ^= 1
        report_dict['execution']['signals'][5]['target'] = float(report_dict['execution']['tape'][5]['prediction'])

    _rewrite_report(root, report, drift, 'retained predictions')

    def reprice(report_dict):
        report_dict['execution']['contract']['policy']['fee_bps'] = 999.0
        report_dict['execution']['assessment_digest'] = contract_digest(report_dict['execution']['contract'])

    _rewrite_report(root, report, reprice, 'contract policy')

    def downgrade(report_dict):
        report_dict['schema_version'] = 1
        del report_dict['execution']

    _rewrite_report(root, report, downgrade, 'schema for pinned')
    flat = trade_execution(with_predictions(replace(spot, initial_equity=equity), [0] * 24), policy)
    assert flat.intents.height == 0
    _rewrite_report(root, report, lambda tampered: tampered['execution'].update(
        ledger={name: getattr(flat, name).to_dicts() for name in ('states', 'intents', 'fills', 'episodes', 'funding')}), 'ledger activity')

    def fabricate(report_dict):
        report_dict['execution']['metrics']['ending_equity']['value'] += 1000.0

    _rewrite_report(root, report, fabricate, 'metrics evidence')

    def misprice(report_dict):
        carried = next(episode for episode in report_dict['execution']['ledger']['episodes'] if episode['closed_at_ns'] is None)
        carried['net_pnl'] += 100.0

    _rewrite_report(root, report, misprice, 'open-episode valuation')

    def backdate(report_dict):
        execution = report_dict['execution']
        for point, signal, row in zip(execution['tape'], execution['signals'], execution['contract']['signal_rows'], strict=True):
            point['available_at_ns'] -= 3600_000_000_000
            signal['available_at_ns'] -= 3600_000_000_000
            row['available_at_ns'] -= 3600_000_000_000
        for intent in execution['ledger']['intents']:
            intent['requested_at_ns'] -= 3600_000_000_000
        execution['assessment_digest'] = contract_digest(execution['contract'])

    _rewrite_report(root, report, backdate, 'availability floor')

    def reprice_fills(report_dict):
        execution = report_dict['execution']
        for fill in execution['ledger']['fills']:
            fill['reference_price'] *= 2.0
            fill['slippage'] = fill['quantity_delta'] * (fill['fill_price'] - fill['reference_price'])
        slip = sum(fill['slippage'] for fill in execution['ledger']['fills'])
        execution['metrics']['slippage']['value'] = slip
        execution['metrics']['gross_pnl']['value'] = (execution['metrics']['ending_equity']['value'] - execution['initial_equity']
                                                      + execution['metrics']['fees']['value'] + slip - execution['metrics']['funding_pnl']['value'])

    _rewrite_report(root, report, reprice_fills, 'fill reference')

    def waive_costs(report_dict):
        report_dict['execution']['ledger']['fills'][0]['fee'] += 1.0
        report_dict['execution']['ledger']['fills'][0]['slippage'] += 1.0
        report_dict['execution']['metrics']['fees']['value'] += 1.0
        report_dict['execution']['metrics']['slippage']['value'] += 1.0
        report_dict['execution']['metrics']['gross_pnl']['value'] += 2.0

    _rewrite_report(root, report, waive_costs, 'fill costs')

    def trim_closed(report_dict):
        shuttered = next(episode for episode in report_dict['execution']['ledger']['episodes'] if episode['closed_at_ns'] is not None)
        shuttered['net_pnl'] += 10.0

    _rewrite_report(root, report, trim_closed, 'closed-episode valuation')

    def reassign(report_dict):
        fills = report_dict['execution']['ledger']['fills']
        intents = report_dict['execution']['ledger']['intents']
        fills[0]['intent_id'] = intents[1]['intent_id']

    _rewrite_report(root, report, reassign, 'fill request order')

    def reprice_costs(execution):
        bps = execution['policy']['fee_bps']
        for fill in execution['ledger']['fills']:
            fill['fee'] = abs(fill['quantity_delta']) * fill['fill_price'] * bps / BPS
            fill['slippage'] = fill['quantity_delta'] * (fill['fill_price'] - fill['reference_price'])
        fees = sum(fill['fee'] for fill in execution['ledger']['fills'])
        slip = sum(fill['slippage'] for fill in execution['ledger']['fills'])
        execution['metrics']['fees']['value'] = fees
        execution['metrics']['slippage']['value'] = slip
        execution['metrics']['gross_pnl']['value'] = (execution['metrics']['ending_equity']['value'] - execution['initial_equity']
                                                      + fees + slip - execution['metrics']['funding_pnl']['value'])

    def reverse(report_dict):
        execution = report_dict['execution']
        fill = execution['ledger']['fills'][0]
        fill['quantity_delta'] = -fill['quantity_delta']
        fill['fill_price'] = fill['reference_price'] * (1 + math.copysign(execution['policy']['slip_bps'] / BPS, fill['quantity_delta']))
        reprice_costs(execution)

    _rewrite_report(root, report, reverse, 'position transition')

    def orphan(report_dict):
        report_dict['execution']['ledger']['fills'][0]['time_ns'] += 1

    _rewrite_report(root, report, orphan, 'fill state coverage')

    def skim(report_dict):
        report_dict['execution']['ledger']['states'][0]['cash'] += 100.0

    _rewrite_report(root, report, skim, 'accounting anchor')

    def double(report_dict):
        execution = report_dict['execution']
        for row in execution['ledger']['states']:
            row['quantity'] *= 2.0
        for fill in execution['ledger']['fills']:
            fill['quantity_delta'] *= 2.0
        reprice_costs(execution)

    _rewrite_report(root, report, double, 'terminal valuation')

    def resize(report_dict):
        execution = report_dict['execution']
        for row in execution['ledger']['states'][:-1]:
            row['quantity'] *= 2.0
        for fill in execution['ledger']['fills']:
            fill['quantity_delta'] *= 2.0
        reprice_costs(execution)

    _rewrite_report(root, report, resize, 'position sizing')

    def reslip(report_dict):
        execution = report_dict['execution']
        for fill in execution['ledger']['fills']:
            fill['fill_price'] *= 1.5
        reprice_costs(execution)

    _rewrite_report(root, report, reslip, 'fill price')

    def inflate(report_dict):
        execution = report_dict['execution']
        states = execution['ledger']['states']
        states[-1]['equity'] += 1000.0
        ending = states[-1]['equity']
        initial = execution['initial_equity']
        metrics = execution['metrics']
        metrics['ending_equity']['value'] = ending
        metrics['total_return']['value'] = ending / initial - 1
        metrics['net_pnl']['value'] = ending - initial
        metrics['gross_pnl']['value'] = ending - initial + metrics['fees']['value'] + metrics['slippage']['value'] - metrics['funding_pnl']['value']
        running, worst = 0.0, 0.0
        for value in [row['equity'] for row in states]:
            running = max(running, value)
            worst = min(worst, value / max(running, initial) - 1)
        metrics['max_drawdown']['value'] = worst
        for episode in execution['ledger']['episodes']:
            if episode['closed_at_ns'] is None:
                episode['net_pnl'] = ending - episode['entry_equity']

    _rewrite_report(root, report, inflate, 'terminal valuation')

    def inject(report_dict):
        states = report_dict['execution']['ledger']['states']
        moments = {fill['time_ns'] for fill in report_dict['execution']['ledger']['fills']}
        target = next(row for row in states[1:] if row['quantity'] == 0 and row['time_ns'] not in moments)
        target['cash'] += 500.0
        target['equity'] = target['cash'] + target['accrued_funding']

    _rewrite_report(root, report, inject, 'cash reconciliation')

    def credit(report_dict):
        report_dict['execution']['ledger']['states'][-1]['accrued_funding'] += 250.0

    _rewrite_report(root, report, credit, 'funding absence')

    def rebase(report_dict):
        states = report_dict['execution']['ledger']['states']
        moments = {fill['time_ns'] for fill in report_dict['execution']['ledger']['fills']}
        target = next(row for row in states[1:] if row['quantity'] == 0 and row['time_ns'] not in moments)
        target['basis'] += 10.0

    _rewrite_report(root, event_case['reports']['perp'], rebase, 'basis reconciliation')

    def expose(report_dict):
        for row in report_dict['execution']['ledger']['states']:
            row['absolute_exposure'] *= 1.5
        report_dict['execution']['metrics']['avg_absolute_exposure']['value'] *= 1.5

    _rewrite_report(root, report, expose, 'exposure evidence')

    def ghost(report_dict):
        target = next(row for row in report_dict['execution']['ledger']['states'] if row['quantity'] == 0)
        target['absolute_exposure'] = 0.25

    _rewrite_report(root, report, ghost, 'exposure evidence')

    def dent(report_dict):
        states = report_dict['execution']['ledger']['states']
        target = next(row for row in states[1:-1] if row['quantity'] == 0)
        target['equity'] -= 500.0
        initial = report_dict['execution']['initial_equity']
        running, worst = 0.0, 0.0
        for value in [row['equity'] for row in states]:
            running = max(running, value)
            worst = min(worst, value / max(running, initial) - 1)
        report_dict['execution']['metrics']['max_drawdown']['value'] = worst

    _rewrite_report(root, report, dent, 'flat-state valuation')

    def dent2(report_dict):
        execution = report_dict['execution']
        states = execution['ledger']['states']
        fills = execution['ledger']['fills']
        times = [row['time_ns'] for row in states]
        moments = [fill['time_ns'] for fill in fills]
        by_time = {row['time_ns']: row for row in states}
        target = next(by_time[fill['time_ns']] for fill in fills if moments.count(fill['time_ns']) == 1 and times.count(fill['time_ns']) == 1 and by_time[fill['time_ns']]['quantity'] != 0)
        target['equity'] -= 500.0
        initial = execution['initial_equity']
        running, worst = 0.0, 0.0
        for value in [row['equity'] for row in states]:
            running = max(running, value)
            worst = min(worst, value / max(running, initial) - 1)
        execution['metrics']['max_drawdown']['value'] = worst

    _rewrite_report(root, report, dent2, 'fill-state valuation')

    def reenter(report_dict):
        execution = report_dict['execution']
        ending = execution['metrics']['ending_equity']['value']
        target = next(episode for episode in execution['ledger']['episodes'] if episode['closed_at_ns'] is None)
        target['entry_equity'] += 200.0
        target['net_pnl'] = ending - target['entry_equity']

    _rewrite_report(root, report, reenter, 'episode entry')

    def shutter(report_dict):
        execution = report_dict['execution']
        episodes = execution['ledger']['episodes']
        moments = {fill['time_ns'] for fill in execution['ledger']['fills']}
        target = next(episode for episode in episodes if episode['closed_at_ns'] is None)
        target['closed_at_ns'] = next(row['time_ns'] for row in execution['ledger']['states'] if row['time_ns'] not in moments and row['time_ns'] > target['first_fill_ns'])
        closed = [episode for episode in episodes if episode['closed_at_ns'] is not None]
        execution['metrics']['completed_trades']['value'] = float(len(closed))
        execution['metrics']['open_trades']['value'] = float(len(episodes) - len(closed))
        execution['metrics']['episode_win_rate']['value'] = sum(episode['net_pnl'] > 0 for episode in closed) / len(closed)

    _rewrite_report(root, report, shutter, 'episode close')


def _rewrite_report(root, report, mutate, match):
    with history.connection(root / 'manifests/holdout.sqlite3') as db:
        stored = db.execute('SELECT report_path, report_sha256 FROM attempts WHERE id=?', (history.identifier(report['attempt_id']),)).fetchone()
    target = root / stored[0]
    original = target.read_bytes()
    try:
        tampered = copy.deepcopy(report)
        mutate(tampered)
        target.write_bytes(json.dumps(tampered, sort_keys=True, separators=(',', ':')).encode())
        with history.connection(root / 'manifests/holdout.sqlite3') as db:
            db.execute('UPDATE attempts SET report_sha256=? WHERE id=?', (hashlib.sha256(target.read_bytes()).hexdigest(), history.identifier(report['attempt_id'])))
            db.commit()
        with pytest.raises(ValueError, match=match):
            read_holdout_report(root, report['attempt_id'])
    finally:
        target.write_bytes(original)
        with history.connection(root / 'manifests/holdout.sqlite3') as db:
            db.execute('UPDATE attempts SET report_sha256=? WHERE id=?', (stored[1], history.identifier(report['attempt_id'])))
            db.commit()


def _structural_config(**overrides):
    base = {'product': None, 'execution_data_source': None, 'funding': None, 'source_interval_seconds': None, 'prediction_mode': 'binary',
            'take_profit_bps_column': None, 'stop_loss_bps_column': None, 'barrier_features': None}
    base.update(overrides)
    return SimpleNamespace(backtest_config=SimpleNamespace(**base))


def test_event_assessment_refusals(event_case):
    root, reservation = event_case['root'], event_case['reservation']
    sensor = event_case['sensors']['spot']
    locator = _locator(event_case, 'spot')
    with pytest.raises(ValueError, match='round locator'):
        _freeze(event_case, 'spot', round_locator=None)
    cohort = Cohort(experiment_log_path=str(event_case['directories']['spot']), permutation_ids=[sensor.permutation_id])
    cohort.set_members([sensor])
    with pytest.raises(ValueError, match='contract-bearing Cohorts'):
        freeze_holdout_selection(cohort, context_start='2024-01-04', scenarios=[], evaluation_target=NextBarUpTarget(pl.DataFrame(), 'observed_up'), round_locator=locator)
    stripped = copy.copy(sensor)
    vars(stripped)['_trade_contract'] = None
    with pytest.raises(ValueError, match='mismatch refuses freezing'):
        events.freeze_kind(stripped, locator)
    plain = copy.deepcopy(event_case['configs']['spot'])
    del plain['sfd']['manifest']['backtest']
    unconfigured = copy.copy(sensor)
    vars(unconfigured)['_manifest'] = CompiledSFD(plain, project_root=root).manifest()
    with pytest.raises(ValueError, match='mismatch refuses freezing'):
        events.freeze_kind(unconfigured, locator)
    mixed = copy.copy(cohort)
    vars(mixed)['_members'] = [sensor, stripped]
    with pytest.raises(ValueError, match='mismatch refuses freezing'):
        freeze_holdout_selection(mixed, context_start='2024-01-04', scenarios=[], evaluation_target=NextBarUpTarget(pl.DataFrame(), 'observed_up'), round_locator=locator)
    broken = copy.copy(cohort)
    vars(broken)['_members'] = [sensor, 123]
    with pytest.raises(ValueError, match='exact Sensor or Cohort'):
        events.freeze_kind(broken, locator)
    with pytest.raises(ValueError, match='exact Sensor or Cohort'):
        events.freeze_kind(123, locator)
    stale = copy.copy(sensor)
    vars(stale)['_trade_digest'] = 'f' * 64
    with pytest.raises(ValueError, match='digest mismatch after copy'):
        events.execution_key(stale, None)
    with pytest.raises(ValueError, match='mismatch refuses freezing'):
        events.contract_execution(unconfigured, locator)
    config = sensor._get_manifest().backtest_config
    with pytest.raises(ValueError, match='execution data sources'):
        holdout_structural(_structural_config(execution_data_source=object()))
    with pytest.raises(ValueError, match='refuses funding'):
        holdout_structural(_structural_config(funding=object()))
    with pytest.raises(ValueError, match='dynamic barriers'):
        holdout_structural(_structural_config(take_profit_bps_column='tp'))
    with pytest.raises(ValueError, match='declared source intervals'):
        holdout_structural(_structural_config(source_interval_seconds=60))
    with pytest.raises(ValueError, match='this product'):
        holdout_structural(_structural_config(product=SimpleNamespace(kind='xxx')))
    from limen.experiment.manifest_core import DataSourceConfig
    with pytest.raises(ValueError, match='funding and execution data sources'):
        holdout_event_policy(replace(config, execution_data_source=DataSourceConfig(lambda **kwargs: pl.DataFrame(), {})), {})
    barriers = {'take_profit_bps_column': 'tp', 'stop_loss_bps_column': 'sl', 'barrier_features': DataSourceConfig(build_search_strategy, {})}
    with pytest.raises(ValueError, match='dynamic barriers'):
        holdout_event_policy(replace(config, **barriers), {})
    with pytest.raises(ValueError, match='declared source intervals'):
        holdout_event_policy(replace(config, source_interval_seconds=60), {})
    with pytest.raises(ValueError, match='Unsupported product accounting'):
        holdout_event_policy(replace(config, product=replace(config.product, kind='xxx')), {})
    relaxed = SimpleNamespace(product=None, execution_data_source=None, source_interval_seconds=None, take_profit_bps_column=None,
                              stop_loss_bps_column=None, barrier_features=None)
    guessed = SimpleNamespace(funding=None, prediction_mode='binary')
    with patch('limen.experiment._resolve_trade_policy.resolve_trade_policy', return_value=guessed), pytest.raises(ValueError, match='this product'):
        holdout_event_policy(relaxed, {})
    before = len(_attempts(root))
    frozen = _freeze(event_case, 'spot', scenarios=[dict(_SCENARIO)])
    with pytest.raises(ValueError, match='refuses scenario declarations'):
        final_holdout_report(frozen, project_root=root, reservation_id=reservation)
    assert len(_attempts(root)) == before


def _drifted_pin(event_case, mutate):
    from limen.experiment._holdout_report import digest
    frozen = _freeze(event_case, 'spot')
    spec = frozen_parts(frozen)[1]
    mutate(spec['members'][0]['execution'])
    object.__setattr__(frozen, '_spec_digest', digest(spec))
    return frozen


def test_event_equity_only_drift_refused(event_case):
    root, reservation = event_case['root'], event_case['reservation']
    before = len(_attempts(root))
    frozen = _drifted_pin(event_case, lambda pin: pin.update(initial_equity=9999.0))
    with pytest.raises(ValueError, match='frozen equity disagrees'):
        final_holdout_report(frozen, project_root=root, reservation_id=reservation)
    assert len(_attempts(root)) == before
    policy = copy.deepcopy(event_case['reports']['spot']['execution']['policy'])
    policy['fee_bps'] = 999.0
    frozen = _drifted_pin(event_case, lambda pin: pin.update(policy=policy))
    with pytest.raises(ValueError, match='frozen policy disagrees'):
        final_holdout_report(frozen, project_root=root, reservation_id=reservation)
    assert len(_attempts(root)) == before
    frozen = _drifted_pin(event_case, lambda pin: pin.update(research_digest='0' * 64))
    with pytest.raises(ValueError, match='frozen digest disagrees'):
        final_holdout_report(frozen, project_root=root, reservation_id=reservation)
    assert len(_attempts(root)) == before


def test_event_templated_params_survive_to_execution(event_case):
    report = event_case['reports']['perp']
    assert event_case['rounds']['perp']['round_params']['equity'] == 25000.0
    execution = report['execution']
    assert execution['initial_equity'] == 25000.0
    assert execution['ledger']['states'][0]['equity'] == 25000.0
    assert execution['contract']['initial_conditions'] == {'equity': 25000.0, 'quantity': 0.0, 'currency': 'USDT'}
    assert report['spec']['members'][0]['execution']['initial_equity'] == 25000.0
    assert read_holdout_report(event_case['root'], report['attempt_id'])['schema_version'] == 2


def test_event_bare_parquet_clock_chain(event_case):
    report = event_case['reports']['spot']
    execution = report['execution']
    names = {field['name'] for field in report['source_schema']}
    assert names & {'start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns', 'base_interval'} == set()
    for point, signal in zip(execution['tape'], execution['signals'], strict=True):
        assert point['available_at_ns'] is not None
        assert point['available_at_ns'] - point['open_ns'] == 3600_000_000_000
        assert signal['available_at_ns'] == point['available_at_ns']
    frame = pl.DataFrame({'datetime': pl.Series(['2024-01-01T00:00:00Z', '2024-01-01T01:00:00Z'], dtype=pl.Datetime(time_unit='us', time_zone='UTC')),
        'open': [1.0, 1.0], 'high': [1.0, 1.0], 'low': [1.0, 1.0], 'close': [1.0, 1.0], 'start_ns': [0, 1], 'end_ns': [1, 2]})
    with pytest.raises(ValueError, match='all four metadata columns'):
        normalize_observations(frame, interval_seconds=3600)


def test_event_partition_bounds_match_allocation(event_case):
    report = event_case['reports']['spot']
    execution = report['execution']
    planned = report['planned']
    partition = execution['contract']['partition']
    assert partition == [planned['context_start_ns'], planned['end_ns']]
    opens = [point['open_ns'] for point in execution['tape']]
    assert opens[0] == planned['start_ns']
    assert opens[-1] + 3600_000_000_000 == planned['end_ns']
    for row in (*execution['ledger']['states'], *execution['ledger']['fills']):
        assert partition[0] <= row['time_ns'] <= partition[1]
    for signal in execution['signals']:
        assert partition[0] < signal['available_at_ns'] <= partition[1]
    rows = [row for row in _attempts(event_case['root']) if (row[2], row[3], row[4]) == (planned['context_start_ns'], planned['start_ns'], planned['end_ns'])]
    assert rows and all(row[0] == 'completed' for row in rows)


def _tampered(sensor, mutate, refresh_digest=False):
    twin = copy.copy(sensor)
    contract = copy.deepcopy(vars(sensor)['_trade_contract'])
    mutate(contract)
    vars(twin)['_trade_contract'] = contract
    if refresh_digest:
        vars(twin)['_trade_digest'] = contract_digest(contract)
    return twin


def test_event_round_locator_binding(event_case):
    sensor = event_case['sensors']['spot']
    directory = event_case['directories']['spot']
    with pytest.raises(ValueError, match='missing from recorded rounds'):
        _freeze(event_case, 'spot', round_locator={'experiment_dir': str(directory), 'round_id': 'missing'})
    path = _round_storage.round_path(directory)
    forged = copy.deepcopy(event_case['rounds']['spot'])
    forged['round_id'] = 'forged-digest-f18a'
    forged['_round_index'] = event_case['rounds']['spot']['_round_index'] + 1
    forged['trade_contract_digest'] = 'f' * 64
    _round_storage.append_record(path, forged)
    with pytest.raises(ValueError, match='disagrees with the recorded round'):
        _freeze(event_case, 'spot', round_locator={'experiment_dir': str(directory), 'round_id': 'forged-digest-f18a'})
    forged['round_id'] = 'forged-body-f18a'
    forged['_round_index'] += 1
    forged['trade_contract_digest'] = event_case['rounds']['spot']['trade_contract_digest']
    forged['trade_contract']['rules']['terminal'] = 'close_all'
    _round_storage.append_record(path, forged)
    with pytest.raises(ValueError, match='body disagrees'):
        _freeze(event_case, 'spot', round_locator={'experiment_dir': str(directory), 'round_id': 'forged-body-f18a'})
    terminal = _tampered(sensor, lambda contract: contract['rules'].update(terminal='close_all'))
    with pytest.raises(ValueError, match='mark_without_close'):
        freeze_holdout_selection(terminal, context_start='2024-01-04', scenarios=[], evaluation_target=NextBarUpTarget(pl.DataFrame(), 'observed_up'), round_locator=_locator(event_case, 'spot'))
    equity = _tampered(sensor, lambda contract: contract['initial_conditions'].update(equity=1.0))
    with pytest.raises(ValueError, match='initial conditions disagree'):
        freeze_holdout_selection(equity, context_start='2024-01-04', scenarios=[], evaluation_target=NextBarUpTarget(pl.DataFrame(), 'observed_up'), round_locator=_locator(event_case, 'spot'))
    digest = copy.copy(sensor)
    vars(digest)['_trade_digest'] = '0' * 64
    with pytest.raises(ValueError, match='digest disagrees with the Sensor record'):
        freeze_holdout_selection(digest, context_start='2024-01-04', scenarios=[], evaluation_target=NextBarUpTarget(pl.DataFrame(), 'observed_up'), round_locator=_locator(event_case, 'spot'))
    crossing = _tampered(sensor, lambda contract: contract['partition'].__setitem__(1, history.utc_day('2024-02-01')), refresh_digest=True)
    with pytest.raises(ValueError, match='crosses the test window'):
        freeze_holdout_selection(crossing, context_start='2024-01-04', scenarios=[], evaluation_target=NextBarUpTarget(pl.DataFrame(), 'observed_up'), round_locator=_locator(event_case, 'spot'))
    drifted = copy.copy(event_case['sensors']['perp'])
    vars(drifted)['_round_params'] = {**vars(drifted)['_round_params'], 'fee': 999.0}
    with pytest.raises(ValueError, match='disagrees with its recorded contract'):
        freeze_holdout_selection(drifted, context_start='2024-01-04', scenarios=[], evaluation_target=NextBarUpTarget(pl.DataFrame(), 'observed_up'), round_locator=_locator(event_case, 'perp'))
    clone = owned_copy(sensor)
    assert vars(clone)['_trade_digest'] == contract_digest(vars(clone)['_trade_contract'])


def test_event_mode_invariant_refusals(event_case):
    sensor = event_case['sensors']['spot']
    config = sensor._get_manifest().backtest_config
    exposed = replace(config, prediction_mode='target_exposure')
    with pytest.raises(ValueError, match='non-binary prediction mode'):
        holdout_event_policy(exposed, {})
    with pytest.raises(ValueError, match='non-binary prediction mode'):
        holdout_structural(SimpleNamespace(backtest_config=exposed))
    policy = holdout_event_policy(config, dict(sensor._round_params))
    with pytest.raises(ValueError, match='output mode does not match'):
        finish_trade_result({'_trade_policy': policy}, {'_preds': [1], '_prediction_mode': 'target_exposure'})
    assert event_case['reports']['spot']['execution']['policy']['prediction_mode'] == 'binary'


def test_event_observed_clock_durable_failures(event_case):
    root, reservation = event_case['root'], event_case['reservation']
    frozen = _freeze(event_case, 'spot')
    before = len(_attempts(root))
    real_tape = assessment._tape

    def shifted(selection, raw, start, failures):
        tape, witnesses = real_tape(selection, raw, start, failures)
        witnesses[0]['available_at_ns'] += 1
        return tape, witnesses

    with patch.object(assessment, '_tape', shifted), pytest.raises(ValueError, match='availability disagrees'):
        final_holdout_report(frozen, project_root=root, reservation_id=reservation, end='2024-01-06')

    def forged(selection, raw, start, failures):
        tape, witnesses = real_tape(selection, raw, start, failures)
        witnesses[0]['trade_contract_digest'] = '0' * 64
        return tape, witnesses

    with patch.object(assessment, '_tape', forged), pytest.raises(ValueError, match='frozen contract'):
        final_holdout_report(frozen, project_root=root, reservation_id=reservation, end='2024-01-06')
    real_inputs = assessment._contract_inputs

    def shifted_bounds(*args, **kwargs):
        inputs, mapping = real_inputs(*args, **kwargs)
        return replace(inputs, partition_start_ns=0), mapping

    with patch.object(assessment, '_contract_inputs', shifted_bounds), pytest.raises(ValueError, match='disagrees with its allocation'):
        final_holdout_report(frozen, project_root=root, reservation_id=reservation, end='2024-01-06')
    real_execute = assessment.trade_execution

    def late_close(bound, policy):
        ledger = real_execute(bound, policy)
        extra = ledger.states.head(1).with_columns(pl.lit(bound.partition_end_ns + 1).alias('time_ns'))
        return replace(ledger, states=pl.concat([ledger.states, extra]))

    with patch.object(assessment, 'trade_execution', late_close), pytest.raises(ValueError, match='past its terminal bound'):
        final_holdout_report(frozen, project_root=root, reservation_id=reservation, end='2024-01-06')

    def bad_digest(bound, policy):
        return replace(real_execute(bound, policy), contract_digest='0' * 64)

    with patch.object(assessment, 'trade_execution', bad_digest), pytest.raises(ValueError, match='disagrees with executed plan'):
        final_holdout_report(frozen, project_root=root, reservation_id=reservation, end='2024-01-06')
    signals = pl.DataFrame({'row_id': ['a', 'b'], 'available_at_ns': [2, 3], 'target': [1.0, 0.0]})
    tiny = TradeInputs(10000.0, 0, 4, signals, signals, None, ())
    mapping = pl.DataFrame({'datetime': pl.Series(['2024-01-01T00:00:00Z', '2024-01-01T01:00:00Z'], dtype=pl.Datetime(time_unit='us', time_zone='UTC')),
        'row_id': ['a', 'b'], 'available_at_ns': [2, 3]})
    witnesses = [{'available_at_ns': 2, 'trade_contract_digest': 'd', 'prediction': 1}] * 3
    with pytest.raises(ValueError, match='disagrees with executed signals'):
        assessment._contract_tape(witnesses, mapping, tiny, 'd')
    failed = _attempts(root)[before:]
    assert len(failed) == 5
    assert all(row[0] == 'failed' for row in failed)
    assert [json.loads(row[1])['stage'] for row in failed] == ['execute'] * 5
    clean = final_holdout_report(frozen, project_root=root, reservation_id=reservation, end='2024-01-06')
    assert clean['state'] == 'completed'
    assert read_holdout_report(root, clean['attempt_id'])['schema_version'] == 2


def test_event_research_availability_cutoff(event_case):
    from limen.experiment._holdout_research import bind_manifest
    root = event_case['root']
    manifest = CompiledSFD(event_case['configs']['spot'], project_root=root).manifest()
    bind_manifest(manifest, root)
    compiled = CompiledSFD(event_case['configs']['spot'], project_root=root)
    params = {key: values[0] for key, values in compiled.params().items()}
    cutoff = history.utc_day('2024-01-05')
    raw = pl.read_parquet(_FIXTURE).head(96)
    starts = raw['datetime'].dt.epoch('ns').to_list()
    step = 3600_000_000_000
    available = [moment + step for moment in starts]
    available[-1] = cutoff + step
    timed = raw.with_columns(pl.Series('start_ns', starts), pl.Series('end_ns', [moment + step for moment in starts]),
        pl.Series('open_available_at_ns', starts), pl.Series('available_at_ns', available))
    with pytest.raises(ValueError, match='crosses the reserved cutoff'):
        manifest.prepare_data(timed, params)
    assert event_case['rounds']['spot']['holdout']['splits']['test']['last_close_ns'] == cutoff


def test_event_rule_based_research_refused(event_case):
    from limen.experiment._holdout_research import bind_manifest, reject_modes
    root = event_case['root']
    rule = copy.deepcopy(event_case['configs']['spot'])
    body = rule['sfd']['manifest']
    body.update(type='rule_based', indicators=[], reference_architecture='limen.sfd.reference_architecture.rule_based.rule_based',
                strategy={'conditions': [{'id': 'up', 'type': 'relative', 'column': 'close', 'operator': '>', 'other_column': 'open'}], 'entry': 'up'})
    for key in ('target', 'scaler'):
        body.pop(key)
    rule['sfd']['params'] = {'sharpe_std_threshold': [0.5]}
    manifest = CompiledSFD(rule, project_root=root).manifest()
    bind_manifest(manifest, root)
    with pytest.raises(ValueError, match='rule-based event execution'):
        reject_modes(manifest)
    body.pop('backtest')
    clean = CompiledSFD(rule, project_root=root).manifest()
    bind_manifest(clean, root)
    reject_modes(clean)
