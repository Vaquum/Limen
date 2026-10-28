from pathlib import Path
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from limen.backtest._snapshot_execution import snapshot_with_execution
from limen.backtest.backtest_snapshot import backtest_snapshot
from limen.backtest.long_flat_strategy import long_flat_strategy
from limen.backtest._event_plan import event_plan
from limen.backtest.trade_contract import contract_digest, export_trade_contract
from limen.backtest.trade_execution import trade_execution
from tests.test_event_execution_reuse import recorded_inputs


@pytest.fixture(scope='module')
def market():
    return pl.read_parquet(Path(__file__).parent / 'fixtures/spot_1h_20240101_20240116.parquet').head(12)


def _execute(market, entries, exits, **options):
    columns = {**market.head(len(entries)).select('open', 'high', 'low', 'close').to_dict(as_series=False),
               'predictions': entries, 'exit': exits}
    columns['price_change'] = np.asarray(columns['close'], dtype=float) - np.asarray(columns['open'], dtype=float)
    return snapshot_with_execution(columns, exit_col='exit', fee_bps=0.0, slip_bps=0.0, **options)


def test_snapshot_entry_holds_until_independent_exit(market):
    _, result = _execute(market, [1, 0, 0, 0, 0, 0], [False, False, False, True, False, False])
    assert result.pos.tolist() == [0, 1, 1, 1, 0, 0]
    assert result.gross[1:4] == pytest.approx(market['close'].to_numpy()[1:4] / market['close'].to_numpy()[:3] - 1)


def test_snapshot_exit_requires_a_post_closure_false_before_reentry(market):
    _, result = _execute(market, [1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
                         [False, False, True, False, False, True, False, False, False, False])
    assert result.pos.tolist() == [0, 1, 1, 0, 0, 1, 0, 0, 0, 1]


def test_snapshot_pending_entry_cancellation_and_rule_lag(market):
    _, result = _execute(market, [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                         [False, True, False, False, False, False, False, True, False, False, False, False],
                         execution_lag_bars=3)
    assert result.pos.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]


def test_snapshot_flat_exit_veto_preserves_initial_arming(market):
    _, result = _execute(market, [1, 1, 1, 0], [True, False, False, False])
    assert result.pos.tolist() == [0, 0, 1, 1]


@pytest.mark.parametrize('lag', [0, -1, True, 1.5])
def test_snapshot_explicit_exit_lag_refusals(market, lag):
    with pytest.raises(ValueError, match='positive integer'):
        _execute(market, [1, 0], [False, False], execution_lag_bars=lag)


def test_snapshot_explicit_exit_evidence_and_price_refusals(market):
    for values in ([0, 1], [False, None], [False]):
        with pytest.raises(ValueError, match=r'boolean|length'):
            _execute(market, [1, 0], values)
    with pytest.raises(ValueError, match='finite positive prices'):
        _execute(market.with_columns(pl.when(pl.int_range(pl.len()) == 1).then(None).otherwise(pl.col('close')).alias('close')), [1, 0], [False, False])
    columns = {**market.head(2).to_dict(as_series=False), 'predictions': [1, 0], 'exit': [False, False]}
    columns['price_change'] = np.asarray(columns['close']) - np.asarray(columns['open'])
    with pytest.raises(ValueError, match='default long-flat strategy'):
        backtest_snapshot(columns, exit_col='exit', strategy=lambda *args, **kwargs: long_flat_strategy(*args, **kwargs))


_RULES = {'version': 'limen-rule-exit-v1', 'signal_semantics': 'entry_until_exit',
          'conditions': [{'id': 'exit', 'type': 'threshold', 'column': 'exit_flag', 'operator': '==', 'value': True}], 'exit': 'exit'}


def _event_case(entries, exits, **options):
    inputs, policy = recorded_inputs()
    signals = inputs.signals.with_columns(pl.Series('target', entries, dtype=pl.Float64), pl.Series('exit', exits, dtype=pl.Boolean))
    return replace(inputs, signals=signals), replace(policy, prediction_mode='binary', exit_rules=dict(_RULES), **options)


def test_event_binary_entry_holds_until_exit():
    inputs, policy = _event_case([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [False] * 3 + [True] + [False] * 8)
    ledger = trade_execution(inputs, policy)
    assert ledger.fills.height == 2
    assert ledger.episodes.height == 1
    assert ledger.episodes['closed_at_ns'][0] == inputs.signals['available_at_ns'][3]
    assert ledger.episodes['exit_reason'][0] == 'rule_exit'


def test_event_pending_entry_cancellation_and_rule_exit_lag():
    inputs, policy = _event_case([1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                                [False, True, False, False, False, False, True, False, False, False, False, False],
                                execution_lag_seconds=1800)
    ledger = trade_execution(inputs, policy)
    assert ledger.intents['status'][0] == 'cancelled'
    assert ledger.fills.height == 2
    assert ledger.episodes['first_fill_ns'][0] == inputs.signals['available_at_ns'][5]
    assert ledger.episodes['closed_at_ns'][0] == inputs.signals['available_at_ns'][8]


def test_event_exit_rearm_ignores_threshold_suppression():
    inputs, policy = _event_case([1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
                                [False, False, True, False, False, True, False, False, False, False, False, False],
                                flat_threshold=0.9, signal_change_bps=9999)
    ledger = trade_execution(inputs, policy)
    assert ledger.episodes.height == 3
    assert ledger.episodes['first_fill_ns'].to_list() == [inputs.signals['available_at_ns'][i] for i in (0, 4, 8)]


def test_event_exit_cache_binds_each_ruleset_and_evidence():
    inputs, policy = _event_case([1] * 12, [False] * 3 + [True] + [False] * 8)
    changed = replace(inputs, signals=inputs.signals.with_columns(pl.Series('exit', [False] * 6 + [True] + [False] * 5)))
    changed_policy = replace(policy, exit_rules={**_RULES, 'conditions': [{**_RULES['conditions'][0], 'value': False}]})
    assert event_plan(inputs, policy) is event_plan(changed, changed_policy)
    first, second = trade_execution(inputs, policy), trade_execution(changed, changed_policy)
    assert first.contract_digest != second.contract_digest
    assert first.contract_digest == contract_digest(export_trade_contract(policy, inputs))
    assert second.contract_digest == contract_digest(export_trade_contract(changed_policy, changed))
    assert first.episodes['closed_at_ns'][0] != second.episodes['closed_at_ns'][0]
    assert first.intents['intent_id'][0].split(':')[0] != second.intents['intent_id'][0].split(':')[0]


def test_event_exit_declaration_evidence_refusals():
    inputs, policy = _event_case([1] * 12, [False] * 12)
    for bad, configured in ((replace(inputs, signals=inputs.signals.drop('exit')), policy),
                            (replace(inputs, signals=inputs.signals.with_columns(pl.lit(None, dtype=pl.Boolean).alias('exit'))), policy),
                            (inputs, replace(policy, exit_rules=None))):
        with pytest.raises(ValueError, match=r'Explicit exit'):
            trade_execution(bad, configured)
    with pytest.raises(ValueError, match='binary'):
        replace(policy, prediction_mode='target_exposure')


def _exit_config(*, event=False):
    from tests.test_dynamic_barriers import _yaml
    config = _yaml()
    settings = config['sfd']['manifest']
    settings['backtest'] = {'fee_bps': 5.0, 'slip_bps': 5.0}
    if event:
        settings['backtest'].update(product={'kind': 'cash_spot', 'instrument': 'BTCUSDT', 'base_currency': 'BTC',
                                            'quote_currency': 'USDT', 'quantity_step': 1e-9, 'min_notional': 0}, source_interval_seconds=3600)
    settings['exit_rules'] = {'conditions': [{'id': 'entry', 'name': 'Close below open', 'type': 'relative',
                                             'column': 'close', 'operator': '<', 'other_column': 'open'}], 'exit': 'entry'}
    return config


@pytest.fixture(scope='module')
def prepared_exits():
    from tests.test_dynamic_barriers import recorded_source
    from limen.yaml import CompiledSFD
    raw = recorded_source()
    result = []
    for event in (False, True):
        manifest = CompiledSFD(_exit_config(event=event)).manifest()
        result.append((manifest, raw, manifest.prepare_data(raw, {'risk_span': 6})))
    return result


def test_native_yaml_exit_rule_parity(prepared_exits):
    from copy import deepcopy
    from limen.yaml import CompiledSFD, validate
    for manifest, raw, data in prepared_exits:
        native = deepcopy(manifest)
        definition = _exit_config()['sfd']['manifest']['exit_rules']
        native.exit_rules_config = None
        assert native.with_exit_rules(definition['conditions'], definition['exit']) is native
        other = native.prepare_data(raw, {'risk_span': 6})
        assert other['_exit_evidence'] == data['_exit_evidence']
        assert native.resolve_trade_policy({'risk_span': 6}) == manifest.resolve_trade_policy({'risk_span': 6})
    config = _exit_config()
    config['sfd']['manifest']['exit_rules']['conditions'][0].update(type='sql_expr', expr='close > AVG(close)')
    assert not validate(config).valid
    with pytest.raises(ValueError, match='sql_expr'):
        CompiledSFD(config).manifest()
    from ruamel.yaml import YAML
    example = YAML(typ='safe').load((Path(__file__).resolve().parents[1] / 'docs/examples/rule_exit.yaml').read_text())
    assert validate(example).valid
    assert CompiledSFD(example).manifest().resolve_trade_policy({'exit_bars': 1}) is None
    config = _exit_config()
    config['sfd']['manifest']['exit_rules']['conditions'][0].pop('name')
    assert not validate(config).valid


def test_rule_entry_holds_until_exit(prepared_exits):
    for manifest, _, data in prepared_exits:
        result = manifest.run_model(data, {'risk_span': 6})
        assert len(result['_preds']) == data['test'].height
        assert result['num_trades_test'] == int(result['_preds'][0] == 1) + int(np.sum(np.diff(result['_preds']) > 0))
        assert data['_alignment']['exit_evidence'] == data['_exit_evidence']['test']
        assert data['test'].columns.count('entry') == 1


def test_ml_entry_holds_until_exit():
    from tests.test_experiment_objective import objective_config, _prepared
    from limen.experiment._objective import prepare_objective
    config = objective_config(calibrated=False)
    config['sfd']['manifest']['exit_rules'] = _exit_config()['sfd']['manifest']['exit_rules']
    manifest, params, data = _prepared(config)
    scorer = prepare_objective(data, manifest.architecture_function, None)
    assert 'exit' in scorer.inputs.signals.columns
    result = manifest.run_model(data, params)
    expected = trade_execution(replace(data['_trade_inputs'], signals=data['_trade_inputs'].signals.with_columns(pl.Series('target', result['_preds'].astype(float)))), data['_trade_policy'])
    assert result['backtest_total_return'] == expected.metrics['total_return']
    assert scorer(None, np.zeros(scorer.inputs.signals.height)) == 0


def test_exit_priority_pending_entry_and_rearm():
    from limen.backtest._rule_exit_state import replace_pending
    from types import SimpleNamespace
    priorities = {'signal': 0, 'rule_exit': 1, 'time_stop': 2, 'stop_loss': 3, 'take_profit': 3}
    for old, old_rank in priorities.items():
        for new, new_rank in priorities.items():
            pending = {'reason': old, 'status': 'pending'}
            account = SimpleNamespace(pending=pending, queued_signal={'target': 1})
            assert replace_pending(account, new) == (new_rank > old_rank)
            assert pending['status'] == ('cancelled' if old == 'signal' else 'replaced') if new_rank > old_rank else pending['status'] == 'pending'


def test_barriers_timeout_lag_and_costs(market):
    inputs, policy = _event_case([1] * 12, [False] * 5 + [True] + [False] * 6)
    ledger = trade_execution(inputs, replace(policy, execution_lag_seconds=1800, max_holding_seconds=900))
    assert ledger.episodes.height
    assert ledger.episodes['exit_reason'][0] == 'time_stop'
    assert 'rule_exit' not in ledger.episodes['exit_reason'].to_list()
    _, free = _execute(market, [1, 0, 0, 0], [False, False, True, False])
    columns = {**market.head(4).to_dict(as_series=False), 'predictions': [1, 0, 0, 0], 'exit': [False, False, True, False]}
    columns['price_change'] = np.asarray(columns['close']) - np.asarray(columns['open'])
    _, charged = snapshot_with_execution(columns, exit_col='exit', fee_bps=10, slip_bps=5)
    assert np.all(charged.net <= free.net)
    assert charged.net.sum() < free.net.sum()
    anchor = float(market['close'][0])
    stop = anchor * (1 - 1 / 10000)
    take = anchor * (1 + 1 / 10000)
    assert float(market['low'][1]) < stop < take < float(market['high'][1])
    _, stopped = _execute(market, [1] * 4, [False, True, False, False], take_profit_bps=1, stop_loss_bps=1)
    assert stopped.pos.tolist() == [0, 1, 0, 0]
    assert stopped.gross[1] == pytest.approx(min(float(market['open'][1]), stop) / anchor - 1)
    inputs, policy = _event_case([1] * 12, [False] * 2 + [True] + [False] * 9)
    forced = trade_execution(inputs, replace(policy, take_profit_bps=1, stop_loss_bps=1))
    assert forced.episodes.height == 1
    assert forced.episodes['exit_reason'][0] == 'stop_loss'
    assert forced.fills.height == 2


def test_exit_evidence_is_causal_and_private(prepared_exits):
    from copy import deepcopy
    from limen.experiment._exit_rules import ExitPreparation
    manifest, _, data = prepared_exits[0]
    native = deepcopy(manifest)
    native.with_exit_rules([{'id': 'up', 'type': 'slope', 'column': 'close', 'lookback': 2},
                            {'id': 'not_up', 'operator': 'not', 'operands': ['up']}], 'not_up')
    source = prepared_exits[0][1].filter(pl.col('datetime').is_between(data['_alignment']['first_test_datetime'], data['_alignment']['last_test_datetime'])).with_row_index('__backtest_source_row__')
    preparation = ExitPreparation(native, {}, None)
    preparation.capture(source)
    before = source.clone()
    selected = source.filter(pl.col('__backtest_source_row__') != 2)
    preparation.frames *= 3
    preparation.select([selected] * 3)
    assert preparation.selected[0]['rows'][0]['exit'] is True
    assert source.equals(before)
    full = ExitPreparation(native, {}, None)
    full.capture(source)
    full.frames *= 3
    full.select([source] * 3)
    assert preparation.selected[0]['rows'] == [row for row in full.selected[0]['rows'] if row['row_id'] != '2']


def test_exit_recording_resume_and_walk_forward(prepared_exits, tmp_path):
    import json
    from copy import deepcopy
    from limen.experiment._exit_rules import admit_run, declaration
    from limen.experiment._recording import project_record
    from tests.test_objective_resume import _loop, _run, _artifact_bytes
    for manifest, _, data in prepared_exits:
        record = {**data['_alignment'], 'preds': [1] * data['test'].height}
        for alignment, predictions in ((True, False), (False, True), (False, False), (True, True)):
            policy = {'alignment': alignment, 'predictions': predictions, 'trade_ledger': False, 'fold_details': True}
            projected = project_record({**record, 'folds': [record]}, policy)
            assert ('exit_evidence' in projected) == alignment
            assert ('preds' in projected) == predictions
            assert ('exit_evidence' in projected['folds'][0]) == alignment
            assert ('exit_rules' in projected or 'trade_contract' in projected)
        metadata = {'exit_rules_declaration': declaration(manifest)}
        admit_run(manifest, metadata, resume=True, factorize=False)
        changed = deepcopy(manifest)
        changed.exit_rules_config = None
        with pytest.raises(ValueError, match='declaration'):
            admit_run(changed, metadata, resume=True, factorize=False)
    config = _exit_config(event=True)
    loop = _loop(config, tmp_path)
    _run(loop)
    metadata = json.loads((tmp_path / 'metadata.json').read_text())
    assert metadata['exit_rules_declaration']['signal_semantics'] == 'entry_until_exit'
    before = _artifact_bytes(tmp_path)
    changed = deepcopy(config)
    changed['sfd']['manifest']['exit_rules']['conditions'][0]['operator'] = '>'
    with pytest.raises(ValueError, match='declaration'):
        _run(_loop(changed, tmp_path), resume=True)
    assert _artifact_bytes(tmp_path) == before
    from tests.test_walk_forward_uel import _config as fold_config, _bars, _loop as fold_loop, _run as fold_run
    from tests.test_record_model_outputs import _records
    folds = fold_config()
    folds['sfd']['manifest']['exit_rules'] = _exit_config()['sfd']['manifest']['exit_rules']
    fold_path = tmp_path / 'folds'
    fold_run(fold_loop(folds, _bars(), fold_path), n_permutations=1)
    stored = _records(fold_path)
    assert len(stored[0]['folds']) == 2
    assert all('exit_evidence' in fold for fold in stored[0]['folds'])
    from tests.test_parallel_admission import run as parallel_run
    parallel_path = tmp_path / 'parallel'
    parallel_run(_exit_config(event=True), parallel_path, workers=2)
    assert _records(parallel_path)[0]['exit_evidence']['rows']
    parallel_run(_exit_config(event=True), parallel_path, workers=2, resume=True)


def test_exit_replay_and_plan_cache_binding(prepared_exits):
    from copy import deepcopy
    from limen.experiment._exit_rules import verify_replay
    from limen.log._snapshot_backtest_round import snapshot_backtest_round
    from limen.log._experiment_backtest_results import _prepare_snapshot_backtest_input
    from types import SimpleNamespace
    for manifest, raw, data in prepared_exits:
        result = manifest.run_model(data, {'risk_span': 6})
        saved = data['_alignment']
        verify_replay(saved, data)
        corrupted = deepcopy(saved)
        corrupted['exit_evidence']['rows'].reverse()
        with pytest.raises(ValueError, match='evidence'):
            verify_replay(corrupted, data)
        log = SimpleNamespace(manifest=manifest, data=raw, round_params=[{'risk_span': 6}], preds=[result['_preds']], _alignment=[saved])
        metrics = snapshot_backtest_round(log, 0, _prepare_snapshot_backtest_input)
        assert 'total_return' in metrics if manifest.resolve_trade_policy({'risk_span': 6}) is not None else len(metrics) == 20
        corrupted = {key: value for key, value in saved.items() if key != 'exit_evidence'}
        log._alignment = [corrupted]
        with pytest.raises(ValueError, match='evidence'):
            snapshot_backtest_round(log, 0, _prepare_snapshot_backtest_input)


def test_unsupported_exit_combinations_fail(prepared_exits):
    from copy import deepcopy
    from limen.experiment._exit_rules import admit_run, finish_snapshot
    from limen.yaml import validate
    for manifest, _, data in prepared_exits:
        with pytest.raises(ValueError, match='factorize'):
            admit_run(manifest, {}, resume=False, factorize=True)
        changed = deepcopy(manifest)
        changed.with_exit_rules([{'id': 'missing', 'type': 'threshold', 'column': 'absent', 'operator': '>', 'value': 0}], 'missing')
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            changed.prepare_data(prepared_exits[0][1], {'risk_span': 6})
        with pytest.raises(ValueError, match='binary'):
            finish_snapshot(data, {'_preds': np.ones(data['test'].height), '_prediction_mode': 'target_exposure'})
        with pytest.raises(ValueError, match='predictions'):
            finish_snapshot(data, {'backtest_return_bps_mean': 10.0})
    config = _exit_config()
    config['uel']['factorize'] = True
    assert not validate(config).valid


def test_unconfigured_results_are_unchanged():
    from tests.test_event_execution_reuse import reference_trade_execution, assert_ledgers
    from limen.backtest.execution_events import SIGNAL_COLUMNS
    inputs, policy = recorded_inputs()
    assert 'exit_rules' not in export_trade_contract(policy, inputs)['policy']
    assert SIGNAL_COLUMNS == ('row_id', 'available_at_ns', 'target')
    assert_ledgers(trade_execution(inputs, policy), reference_trade_execution(inputs, policy))
    frozen = replace(inputs, sources=tuple(replace(source, identity='recorded-test-fixture') for source in inputs.sources))
    contract = export_trade_contract(policy, frozen)
    # Captured from unchanged main f91e4bc8, before exit policy serialization changed.
    assert contract_digest(contract) == 'ea7a58000e444cf5a495215551bf5e6498bea7114a84b4e5657ffe99c257b654'
    from limen.experiment._prepare_trade_context import validate_inference_contract
    validate_inference_contract(policy, policy.prediction_mode, contract, {'trade_contract_digest': contract_digest(contract)})
