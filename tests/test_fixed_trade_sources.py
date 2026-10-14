import copy
import hashlib
import json
from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from threadpoolctl import threadpool_limits

from limen.data import HistoricalData
from limen.experiment import UniversalExperimentLoop
from limen.experiment.manifest_core import DataSourceConfig, MLManifest, RuleBasedManifest
from limen.sfd.reference_architecture import LogRegBinary
from limen.yaml import CompiledSFD, build_search_strategy
from tests.test_record_model_outputs import _config as _binary_config


FIXTURES = Path(__file__).parent / 'fixtures'
FUNDING_FIXTURE = FIXTURES / 'btcusdt_funding_20240101_20240108.json'
SOURCE_CALLS = []
_NS_PER_MS = 1000000
_SLOT_MS = 28800000


def _funding_frame():
    records = json.loads(FUNDING_FIXTURE.read_text())['records']
    slots = [row['fundingTime'] // _SLOT_MS * _SLOT_MS for row in records]
    return pl.DataFrame({
        'event_id': [f"BTCUSDT:{row['fundingTime']}" for row in records],
        'settlement_at': [row['fundingTime'] * _NS_PER_MS for row in records],
        'settlement_slot_ns': [slot * _NS_PER_MS for slot in slots],
        'rate_decimal': [float(row['fundingRate']) for row in records],
        'valuation_price': [float(row['markPrice']) for row in records],
        'schedule_start': [min(slots) * _NS_PER_MS] * len(records),
        'schedule_end': [(max(slots) + _SLOT_MS) * _NS_PER_MS] * len(records),
        'settlement_interval_seconds': [28800.0] * len(records),
        'settlement_phase_utc_seconds': [0.0] * len(records),
    })


def recorded_source(*, role='execution', kline_size=3600, marker=None):
    SOURCE_CALLS.append({'role': role, 'kline_size': kline_size, 'marker': copy.deepcopy(marker)})
    if role == 'funding':
        return _funding_frame()
    return pl.read_parquet(FIXTURES / 'spot_1h_20240101_20241231.parquet', n_rows=192)


@pytest.fixture(scope='module')
def recorded_learning_bars():
    return pl.read_parquet(FIXTURES / 'spot_1h_20240101_20241231.parquet', n_rows=168)


@pytest.fixture(autouse=True)
def _source_calls_and_native_limit(monkeypatch):
    SOURCE_CALLS.clear()
    monkeypatch.setattr(HistoricalData, 'get_any_file', staticmethod(recorded_source))
    with threadpool_limits(limits=1):
        yield


def _config(*, sources=True, walk_forward=False):
    config = _binary_config()
    config['metadata']['name'] = 'fixed_trade_sources'
    manifest = config['sfd']['manifest']
    method = 'limen.data.HistoricalData.get_any_file'
    manifest['data_source'] = {'method': method, 'params': {'role': 'learning', 'kline_size': 3600}}
    manifest['split_dates'] = {
        'train_start': '2024-01-01', 'train_end': '2024-01-04',
        'val_start': '2024-01-04', 'val_end': '2024-01-06',
        'test_start': '2024-01-06', 'test_end': '2024-01-08',
    }
    manifest['backtest'] = {
        'product': {'kind': 'linear_perpetual', 'instrument': 'BTCUSDT',
                    'base_currency': 'BTC', 'quote_currency': 'USDT',
                    'quantity_step': 1e-9, 'min_notional': 0.0},
        'fee_bps': '{fee_bps}', 'slip_bps': 2.0,
    }
    if sources:
        manifest['backtest']['execution_data_source'] = {
            'method': method, 'params': {'role': 'execution', 'kline_size': 3600},
        }
        manifest['backtest']['funding'] = {
            'preset': 'binance_btcusdt',
            'data_source': {'method': method, 'params': {'role': 'funding'}},
        }
    config['sfd']['params'] = {'C': [0.3], 'fee_bps': [3.0, 7.0], 'n_jobs': [1],
                             'random_state': [42], 'max_iter': [200], 'tol': [1e-7]}
    config['uel'].update(n_permutations=2, checkpoint_interval=1,
                         search_strategy={'type': 'grid'})
    if walk_forward:
        del manifest['split_dates']
        manifest['split_walk_forward'] = {'n_folds': 2, 'test_bars': 24,
                                         'purge_bars': 2, 'embargo_bars': 1, 'anchored': True}
    return config


class _NativeSFD:
    def __init__(self, config):
        self.__name__ = __name__
        self.compiled = CompiledSFD(config)
        self.native_manifest = self.compiled.manifest()
        backtest = self.native_manifest.backtest_config
        self.native_manifest.backtest_config = replace(
            backtest,
            execution_data_source=DataSourceConfig(recorded_source, {'role': 'execution', 'kline_size': 3600}),
            funding=replace(backtest.funding, data_source=DataSourceConfig(recorded_source, {'role': 'funding'})),
        )

    def params(self):
        return self.compiled.params()

    def manifest(self):
        return self.native_manifest


def _loop(config, bars, directory, *, native=False, search=True):
    return UniversalExperimentLoop(
        sfd=_NativeSFD(config) if native else CompiledSFD(config), data=bars,
        yaml_reference=config, experiment_dir=directory,
        search_strategy=build_search_strategy(config) if search else None,
        checkpoint_interval=1,
    )


def _run(loop, name='fixed_sources', **kwargs):
    loop.run(name, n_permutations=2, prep_each_round=True, random_search=False,
             progress_bar=False, record_execution=loop._search_strategy is not None,
             record_model_outputs=loop._search_strategy is not None, **kwargs)


def _records(directory):
    return [json.loads(line) for line in (directory / 'round_data.jsonl').read_text().splitlines()]


def _assert_recorded_equal(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_recorded_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_recorded_equal(left, right)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=0.0, abs=0.0, nan_ok=True)
    else:
        assert actual == expected


def _assert_campaigns_equal(first, second, first_path, second_path):
    assert first.experiment_log.drop('execution_time').equals(second.experiment_log.drop('execution_time'))
    _assert_recorded_equal(_records(first_path), _records(second_path))
    if first.manifest.split_walk_forward is not None:
        assert first.fold_results.equals(second.fold_results)
        assert pl.read_parquet(first_path / 'trial_returns.parquet').equals(
            pl.read_parquet(second_path / 'trial_returns.parquet'),
        )
    else:
        rows = _records(first_path)
        assert [row['_round_index'] for row in rows] == [0, 1]
        assert len({row['round_id'] for row in rows}) == 2
        assert all(row['trade_contract']['policy']['funding']['approximation'] == 'recorded_exact' for row in rows)
        assert all(row['trade_ledger']['funding'] for row in rows)


def test_recorded_funding_fixture_preserves_official_rates_and_timestamps():
    fixture = json.loads(FUNDING_FIXTURE.read_text())
    records = fixture['records']
    assert len(records) == 22
    canonical = json.dumps(records, separators=(',', ':'), ensure_ascii=False).encode()
    assert hashlib.sha256(canonical).hexdigest() == fixture['records_sha256']
    assert fixture['response_sha256'] == 'ef43c5d7b2dbc90d697d7496f94c4c0885f20fde674da3a9201f9298986366a5'
    assert fixture['source_url'].startswith('https://fapi.binance.com/fapi/v1/fundingRate?')
    frame = _funding_frame()
    assert frame['rate_decimal'].to_list() == [float(row['fundingRate']) for row in records]
    assert frame['valuation_price'].to_list() == [float(row['markPrice']) for row in records]
    assert frame['settlement_at'].to_list() == [row['fundingTime'] * _NS_PER_MS for row in records]
    assert frame['settlement_slot_ns'].diff().drop_nulls().unique().to_list() == [_SLOT_MS * _NS_PER_MS]
    assert frame['settlement_at'].ge(frame['settlement_slot_ns']).all()
    assert (frame['settlement_at'] - frame['settlement_slot_ns']).max() == _NS_PER_MS


@pytest.mark.parametrize('native', (False, True))
def test_fixed_secondary_sources_match_legacy_recorded_outputs(recorded_learning_bars, tmp_path, monkeypatch, native):
    config = _config()
    fits = []
    original_train = LogRegBinary.train

    def train(model, data, **params):
        fits.append((model, data['x_train']))
        return original_train(model, data, **params)

    monkeypatch.setattr(LogRegBinary, 'train', train)
    fixed_path, legacy_path = tmp_path / 'fixed', tmp_path / 'legacy'
    fixed = _loop(config, recorded_learning_bars, fixed_path, native=native)
    _run(fixed)
    assert Counter(row['role'] for row in SOURCE_CALLS) == {'execution': 1, 'funding': 1}
    assert not hasattr(fixed.manifest, '_frozen_trade_sources')
    SOURCE_CALLS.clear()
    legacy = _loop(config, recorded_learning_bars, legacy_path, native=native)
    with monkeypatch.context() as baseline:
        baseline.setattr('limen.experiment.experiment_core.fixed_trade_sources', lambda *args: nullcontext())
        _run(legacy)
    assert Counter(row['role'] for row in SOURCE_CALLS) == {'execution': 2, 'funding': 2}
    assert len(fits) == 4
    assert len({id(model) for model, _ in fits}) == 4
    assert len({id(frame) for _, frame in fits}) == 4
    _assert_campaigns_equal(fixed, legacy, fixed_path, legacy_path)


def test_fixed_secondary_sources_reload_on_next_invocation(recorded_learning_bars, tmp_path):
    loop = _loop(_config(), recorded_learning_bars, tmp_path, search=False)
    _run(loop, 'first')
    assert Counter(row['role'] for row in SOURCE_CALLS) == {'execution': 1, 'funding': 1}
    assert not hasattr(loop.manifest, '_frozen_trade_sources')
    _run(loop, 'second')
    assert Counter(row['role'] for row in SOURCE_CALLS) == {'execution': 2, 'funding': 2}
    assert not hasattr(loop.manifest, '_frozen_trade_sources')


@pytest.mark.parametrize('stop', (False, True))
def test_fixed_secondary_sources_cleanup_after_failure_or_stop(recorded_learning_bars, tmp_path, monkeypatch, stop):
    loop = _loop(_config(), recorded_learning_bars, tmp_path / 'interrupted')
    original_train = LogRegBinary.train

    def interrupted_train(model, data, **params):
        if not stop:
            raise RuntimeError('Recorded model failure after source acquisition')
        result = original_train(model, data, **params)
        loop._shutdown_requested = True
        return result

    with monkeypatch.context() as interrupted:
        interrupted.setattr(LogRegBinary, 'train', interrupted_train)
        if stop:
            _run(loop)
            assert loop.experiment_log.height == 1
        else:
            with pytest.raises(RuntimeError, match='Recorded model failure'):
                _run(loop)
    assert Counter(row['role'] for row in SOURCE_CALLS) == {'execution': 1, 'funding': 1}
    assert not hasattr(loop.manifest, '_frozen_trade_sources')
    next_loop = _loop(_config(), recorded_learning_bars, tmp_path / 'next')
    _run(next_loop)
    assert Counter(row['role'] for row in SOURCE_CALLS) == {'execution': 2, 'funding': 2}
    assert not hasattr(next_loop.manifest, '_frozen_trade_sources')


@pytest.mark.parametrize('declaration,source,expected_markers', (
    ('fee_bps', 'execution', ['fee_bps'] * 2),
    ('{fee_bps}', 'funding', [3.0, 7.0]),
    ({'nested': ['{fee_bps}']}, 'execution', [{'nested': [3.0]}, {'nested': [7.0]}]),
    ('context_source', 'execution', ['context_source'] * 2),
    ('{context_source}', 'execution', ['recorded-context'] * 2),
))
def test_dynamic_secondary_sources_keep_whole_pair_per_round(recorded_learning_bars, tmp_path, declaration, source, expected_markers):
    config = _config()
    backtest = config['sfd']['manifest']['backtest']
    params = (backtest['execution_data_source']['params'] if source == 'execution'
              else backtest['funding']['data_source']['params'])
    params['marker'] = declaration
    context = {'context_source': 'recorded-context'} if declaration in ('context_source', '{context_source}') else None
    loop = _loop(config, recorded_learning_bars, tmp_path)
    _run(loop, context_params=context)
    assert Counter(row['role'] for row in SOURCE_CALLS) == {'execution': 2, 'funding': 2}
    assert not hasattr(loop.manifest, '_frozen_trade_sources')
    markers = [row['marker'] for row in SOURCE_CALLS if row['role'] == source]
    assert markers == expected_markers


def test_parameter_override_retains_loading_without_double_evaluation(recorded_learning_bars, tmp_path):
    config = _config()
    calls = []

    def params():
        calls.append('params')
        return copy.deepcopy(config['sfd']['params'])

    loop = _loop(config, recorded_learning_bars, tmp_path, search=False)
    _run(loop, params=params)
    assert calls == ['params']
    assert Counter(row['role'] for row in SOURCE_CALLS) == {'execution': 2, 'funding': 2}
    assert not hasattr(loop.manifest, '_frozen_trade_sources')


def test_absent_secondary_sources_do_not_acquire_data(recorded_learning_bars, tmp_path):
    loop = _loop(_config(sources=False), recorded_learning_bars, tmp_path)
    _run(loop)
    assert SOURCE_CALLS == []
    assert loop.experiment_log.height == 2
    assert not hasattr(loop.manifest, '_frozen_trade_sources')


def test_walk_forward_fixed_secondary_sources_load_once_across_folds(recorded_learning_bars, tmp_path, monkeypatch):
    config = _config(walk_forward=True)
    fixed_path, legacy_path = tmp_path / 'fixed', tmp_path / 'legacy'
    fold_manifests = []
    original_prepare = MLManifest.prepare_data

    def prepare(manifest, raw, params):
        prepared = original_prepare(manifest, raw, params)
        fold_manifests.append((manifest, dict(params), prepared['_trade_context'].partitions[2]))
        return prepared

    with monkeypatch.context() as capture:
        capture.setattr(MLManifest, 'prepare_data', prepare)
        fixed = _loop(config, recorded_learning_bars, fixed_path)
        _run(fixed)
    assert Counter(row['role'] for row in SOURCE_CALLS) == {'execution': 1, 'funding': 1}
    assert not hasattr(fixed.manifest, '_frozen_trade_sources')
    assert len(fold_manifests) == 4
    assert all(manifest is not fixed.manifest for manifest, _, _ in fold_manifests)
    SOURCE_CALLS.clear()
    legacy = _loop(config, recorded_learning_bars, legacy_path)
    with monkeypatch.context() as baseline:
        baseline.setattr('limen.experiment.experiment_core.fixed_trade_sources', lambda *args: nullcontext())
        _run(legacy)
    assert Counter(row['role'] for row in SOURCE_CALLS) == {'execution': 4, 'funding': 4}
    assert fixed.fold_results.height == 4
    _assert_campaigns_equal(fixed, legacy, fixed_path, legacy_path)
    SOURCE_CALLS.clear()
    copied_manifest, params, recorded_partition = fold_manifests[0]
    for call_count in (1, 2):
        prepared = copied_manifest.prepare_data(recorded_learning_bars, params)
        partition = prepared['_trade_context'].partitions[2]
        assert partition.observations.equals(recorded_partition.observations)
        assert partition.funding_events.equals(recorded_partition.funding_events)
        assert partition.sources == recorded_partition.sources
        assert Counter(row['role'] for row in SOURCE_CALLS) == {'execution': call_count, 'funding': call_count}
    assert not hasattr(fixed.manifest, '_frozen_trade_sources')


def test_fixed_secondary_sources_acquisition_is_lazy_and_run_local(recorded_learning_bars, tmp_path):
    from limen.experiment._fixed_trade_sources import fixed_trade_sources

    loop = _loop(_config(), recorded_learning_bars, tmp_path)
    manifest = loop.manifest
    with fixed_trade_sources(manifest, set(loop.params)):
        assert SOURCE_CALLS == []
    assert not hasattr(manifest, '_frozen_trade_sources')
    with fixed_trade_sources(manifest, set(loop.params)):
        provider = manifest._frozen_trade_sources
        first = provider(manifest.backtest_config, {'fee_bps': 3.0})
        second = provider(manifest.backtest_config, {'fee_bps': 7.0})
        assert Counter(row['role'] for row in SOURCE_CALLS) == {'execution': 1, 'funding': 1}
        assert first[0].equals(second[0])
        assert first[1].equals(second[1])
        assert first[0].equals(pl.read_parquet(FIXTURES / 'spot_1h_20240101_20241231.parquet', n_rows=192))
        assert first[1].equals(_funding_frame())
    assert not hasattr(manifest, '_frozen_trade_sources')


def test_existing_frozen_source_tuple_remains_owned_by_its_caller(recorded_learning_bars, tmp_path):
    from limen.experiment._fixed_trade_sources import fixed_trade_sources

    loop = _loop(_config(), recorded_learning_bars, tmp_path)
    manifest = loop.manifest
    frozen = (pl.read_parquet(FIXTURES / 'spot_1h_20240101_20241231.parquet', n_rows=192),
              _funding_frame())
    manifest._frozen_trade_sources = frozen
    params = {name: values[0] for name, values in loop.params.items()}
    with fixed_trade_sources(manifest, set(loop.params)):
        assert manifest._frozen_trade_sources is frozen
        prepared = manifest.prepare_data(recorded_learning_bars, params)
        assert prepared['_trade_context'].partitions[2].funding_events.height > 0
    assert SOURCE_CALLS == []
    assert manifest._frozen_trade_sources is frozen


def test_failed_secondary_acquisition_propagates_without_partial_snapshot(recorded_learning_bars, tmp_path, monkeypatch):
    import limen.experiment._fixed_trade_sources as source_scope

    loop = _loop(_config(), recorded_learning_bars, tmp_path / 'failed')
    original_load = source_scope._load
    original_train = LogRegBinary.train
    fits = []

    def failing_load(source, params):
        if source.params['role'] == 'funding':
            raise RuntimeError('Recorded funding acquisition failed')
        return original_load(source, params)

    def train(model, data, **params):
        fits.append(model)
        return original_train(model, data, **params)

    monkeypatch.setattr(LogRegBinary, 'train', train)
    with monkeypatch.context() as failed:
        failed.setattr(source_scope, '_load', failing_load)
        with pytest.raises(RuntimeError, match='Recorded funding acquisition failed'):
            _run(loop)
    assert fits == []
    assert loop.experiment_log is None or loop.experiment_log.is_empty()
    assert Counter(row['role'] for row in SOURCE_CALLS) == {'execution': 1}
    assert not hasattr(loop.manifest, '_frozen_trade_sources')
    next_loop = _loop(_config(), recorded_learning_bars, tmp_path / 'next')
    _run(next_loop)
    assert len(fits) == 2
    assert Counter(row['role'] for row in SOURCE_CALLS) == {'execution': 2, 'funding': 1}
    assert not hasattr(next_loop.manifest, '_frozen_trade_sources')


class _RecordedSourceReceiver:
    def __init__(self):
        self.calls = 0

    def load(self, **params):
        self.calls += 1
        return recorded_source(**params)


@pytest.mark.parametrize('changed_owner', ('parent', 'copied'))
def test_fixed_source_copy_reuses_snapshot_and_rejects_declaration_drift(recorded_learning_bars, tmp_path, changed_owner):
    from limen.experiment._fixed_trade_sources import fixed_trade_sources

    loop = _loop(_config(), recorded_learning_bars, tmp_path)
    manifest = loop.manifest
    receiver = _RecordedSourceReceiver()
    manifest.backtest_config = replace(
        manifest.backtest_config,
        execution_data_source=DataSourceConfig(receiver.load, {'role': 'execution', 'kline_size': 3600}),
    )
    with fixed_trade_sources(manifest, set(loop.params)):
        first = manifest._frozen_trade_sources(manifest.backtest_config, {'fee_bps': 3.0})
        copied = copy.deepcopy(manifest)
        second = copied._frozen_trade_sources(copied.backtest_config, {'fee_bps': 7.0})
        assert first[0].equals(second[0])
        assert first[1].equals(second[1])
        assert receiver.calls == 1
        assert copied.backtest_config.execution_data_source.method.__self__.calls == 1
        assert Counter(row['role'] for row in SOURCE_CALLS) == {'execution': 1, 'funding': 1}
        changed = manifest if changed_owner == 'parent' else copied
        source = changed.backtest_config.execution_data_source
        changed.backtest_config = replace(
            changed.backtest_config,
            execution_data_source=DataSourceConfig(source.method, {**source.params, 'marker': 'changed'}),
        )
        with pytest.raises(ValueError, match='declaration changed'):
            copied._frozen_trade_sources(copied.backtest_config, {'fee_bps': 7.0})
        assert Counter(row['role'] for row in SOURCE_CALLS) == {'execution': 1, 'funding': 1}
    assert not hasattr(manifest, '_frozen_trade_sources')


class _RecordedSourceCallable:
    def __init__(self):
        self.calls = 0

    def __call__(self, **params):
        self.calls += 1
        return recorded_source(**params)


def test_callable_instance_sources_remain_usable_in_walk_forward(recorded_learning_bars, tmp_path):
    loop = _loop(_config(walk_forward=True), recorded_learning_bars, tmp_path, native=True)
    loader = _RecordedSourceCallable()
    loop.manifest.backtest_config = replace(
        loop.manifest.backtest_config,
        execution_data_source=DataSourceConfig(loader, {'role': 'execution', 'kline_size': 3600}),
    )
    _run(loop)
    assert Counter(row['role'] for row in SOURCE_CALLS) == {'execution': 4, 'funding': 4}
    assert loader.calls == 0
    assert not hasattr(loop.manifest, '_frozen_trade_sources')
    assert loop.fold_results.height == 4
    rows = _records(tmp_path)
    assert [row['_round_index'] for row in rows] == [0, 1]
    assert all(len(row['folds']) == 2 for row in rows)
    for row in rows:
        for fold in row['folds']:
            assert len(fold['preds']) == len(fold['net_returns']) > 0
            assert fold['trade_contract']['policy']['funding']['approximation'] == 'recorded_exact'
            assert fold['trade_ledger']['states']


@pytest.mark.parametrize('method_name,expected_markers', (
    ('prepare_data', [3.0, 7.0]), ('run_model', [None, 3.0]),
))
def test_replaced_manifest_methods_preserve_per_round_source_loading(recorded_learning_bars, tmp_path, monkeypatch, method_name, expected_markers):
    loop = _loop(_config(), recorded_learning_bars, tmp_path)
    manifest = loop.manifest
    original = getattr(manifest, method_name)
    calls = []

    def change_source(params):
        source = manifest.backtest_config.execution_data_source
        manifest.backtest_config = replace(
            manifest.backtest_config,
            execution_data_source=DataSourceConfig(source.method, {**source.params, 'marker': params['fee_bps']}),
        )

    def overridden(data, round_params):
        calls.append(round_params['fee_bps'])
        assert not hasattr(manifest, '_frozen_trade_sources')
        if method_name == 'prepare_data':
            change_source(round_params)
        result = original(data, round_params)
        if method_name == 'run_model':
            change_source(round_params)
        return result

    monkeypatch.setattr(manifest, method_name, overridden)
    _run(loop)
    assert calls == [3.0, 7.0]
    assert Counter(row['role'] for row in SOURCE_CALLS) == {'execution': 2, 'funding': 2}
    assert [row['marker'] for row in SOURCE_CALLS if row['role'] == 'execution'] == expected_markers
    assert [row['_round_index'] for row in _records(tmp_path)] == [0, 1]
    assert not hasattr(manifest, '_frozen_trade_sources')


def test_removing_only_secondary_source_mid_run_fails_loudly(recorded_learning_bars, tmp_path, monkeypatch):
    config = _config()
    del config['sfd']['manifest']['backtest']['funding']
    loop = _loop(config, recorded_learning_bars, tmp_path)
    original_train = LogRegBinary.train
    fits = []

    def train(model, data, **params):
        fitted = original_train(model, data, **params)
        fits.append(model)
        loop.manifest.backtest_config = replace(loop.manifest.backtest_config, execution_data_source=None)
        return fitted

    monkeypatch.setattr(LogRegBinary, 'train', train)
    with pytest.raises(ValueError, match='declaration changed'):
        _run(loop)
    assert len(fits) == 1
    assert Counter(row['role'] for row in SOURCE_CALLS) == {'execution': 1}
    assert [row['_round_index'] for row in _records(tmp_path)] == [0]
    assert not hasattr(loop.manifest, '_frozen_trade_sources')


def test_rule_based_fixed_sources_match_legacy_recorded_outputs(recorded_learning_bars, tmp_path, monkeypatch):
    config = _config()
    manifest = config['sfd']['manifest']
    config['sfd']['manifest'] = {
        'type': 'rule_based',
        **{key: manifest[key] for key in ('data_source', 'split_dates', 'backtest')},
        'strategy': {
            'conditions': [{'id': 'entry', 'name': 'entry', 'type': 'relative',
                            'column': 'close', 'operator': '>', 'other_column': 'open'}],
            'entry': 'entry',
        },
        'reference_architecture': 'limen.sfd.reference_architecture.rule_based',
    }
    config['sfd']['params'] = {'fee_bps': [3.0, 7.0]}
    fixed_path, legacy_path = tmp_path / 'fixed', tmp_path / 'legacy'
    fixed = _loop(config, recorded_learning_bars, fixed_path)
    assert isinstance(fixed.manifest, RuleBasedManifest)
    _run(fixed)
    assert Counter(row['role'] for row in SOURCE_CALLS) == {'execution': 1, 'funding': 1}
    assert not hasattr(fixed.manifest, '_frozen_trade_sources')
    SOURCE_CALLS.clear()
    legacy = _loop(config, recorded_learning_bars, legacy_path)
    with monkeypatch.context() as baseline:
        baseline.setattr('limen.experiment.experiment_core.fixed_trade_sources', lambda *args: nullcontext())
        _run(legacy)
    assert Counter(row['role'] for row in SOURCE_CALLS) == {'execution': 2, 'funding': 2}
    assert fixed.experiment_log.drop('execution_time').equals(legacy.experiment_log.drop('execution_time'))
    rows = _records(fixed_path)
    _assert_recorded_equal(rows, _records(legacy_path))
    assert [row['_round_index'] for row in rows] == [0, 1]
    assert all(row['trade_contract']['policy']['funding']['approximation'] == 'recorded_exact' for row in rows)
