from __future__ import annotations

import copy
import itertools
import json
import random
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy.stats import qmc

from limen.experiment import UniversalExperimentLoop, _round_storage as storage
from limen.experiment.msq import MSQ
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import GridStrategy, RandomStrategy, SobolStrategy, STRATEGY_REGISTRY
from limen.yaml import CompiledSFD, build_search_strategy, validate
from tests.test_factorized_preparation import _config as preparation_config


FIXTURE = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'
SEED = 42


@pytest.fixture(scope='module')
def recorded_bars():
    return pl.read_parquet(FIXTURE, n_rows=192).with_columns(pl.lit(3600).alias('base_interval'))


def _recorded_domain(bars):
    return {
        'volume': bars['volume'].unique(maintain_order=True).head(4).to_list(),
        'close': bars['close'].unique(maintain_order=True).head(8).to_list(),
    }


def _clean(combo):
    return {key: value for key, value in combo.items() if not key.startswith('_')}


def _project(point, params):
    return {key: params[key][int(point[index] * len(params[key]))]
            for index, key in enumerate(sorted(params))}


def _engine(params):
    return qmc.Sobol(d=len(params), scramble=True, bits=30, rng=np.random.default_rng(SEED))


def _config():
    config = preparation_config()
    config['metadata']['name'] = 'sobol_search'
    config['sfd']['manifest']['reference_architecture'] = 'limen.sfd.reference_architecture.logreg_binary'
    config['sfd']['params'] = {
        'lookback': [2, 4, 6, 8], 'quantile': [0.5], 'fee': [4., 9.],
        'C': [0.1, 1.0], 'max_iter': [200], 'n_jobs': [1], 'random_state': [SEED],
    }
    config['uel'].update(n_permutations=6, search_strategy={'type': 'sobol', 'seed': SEED})
    return config


def _prune(log, msq):
    if log.height == 2:
        msq.remove_is('C', 0.1)


def _loop(config, bars, path, *, native=False):
    strategy = (SobolStrategy(ParamDomain(config['sfd']['params']), seed=SEED)
                if native else build_search_strategy(config))
    return UniversalExperimentLoop(
        sfd=CompiledSFD(config), data=bars, experiment_dir=path, yaml_reference=config,
        search_strategy=strategy, checkpoint_interval=1, feedback_interval=1, intra_callback=_prune,
    )


def _run(loop, *, resume=False, factorize=False):
    loop.run('sobol_search', n_permutations=6, prep_each_round=True, progress_bar=False,
             resume=resume, factorize=factorize, record_model_outputs=True, record_execution=True,
             recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True))


def _stop_at_checkpoint(loop, index):
    original = loop._checkpoint

    def stop(*args, **kwargs):
        original(*args, **kwargs)
        if args[4] == index:
            loop._shutdown_requested = True

    loop._checkpoint = stop


def _snapshot(path):
    return {str(file.relative_to(path)): (file.read_bytes(), file.stat().st_mtime_ns)
            for file in path.rglob('*') if file.is_file()}


def _science(path):
    rows = pl.read_csv(path / 'results.csv', infer_schema_length=None).drop('execution_time')
    records = [json.loads(line) for line in storage.text(storage.round_path(path)).splitlines()]
    audit = [json.loads(line) for line in (path / 'audit.jsonl').read_text().splitlines()]
    for entry in audit:
        entry.pop('timestamp')
    return rows, records, audit


@pytest.fixture(scope='module')
def partial_run(tmp_path_factory, recorded_bars):
    path = tmp_path_factory.mktemp('sobol_partial')
    config = _config()
    loop = _loop(config, recorded_bars, path)
    _stop_at_checkpoint(loop, 2)
    _run(loop)
    assert loop.experiment_log.height == 3
    return config, path


def test_sobol_coordinate_projection_and_finite_domain_coverage(recorded_bars):
    params = _recorded_domain(recorded_bars)
    expected = [_project(point, params) for point in _engine(params).random_base2(5)]
    strategy = SobolStrategy(ParamDomain(params), seed=SEED)
    actual = list(MSQ(strategy, strategy.domain, n_permutations=32))
    assert [_clean(combo) for combo in actual] == expected
    assert {tuple(combo[key] for key in sorted(params)) for combo in actual} == set(
        itertools.product(*(params[key] for key in sorted(params))))
    assert strategy.is_finite is False
    assert strategy.get_state()['raw_index'] == strategy.generated_count == 32
    assert all(combo['_search_strategy'] == 'SobolStrategy' for combo in actual)
    assert [combo['_generation_index'] for combo in actual] == list(range(32))
    assert STRATEGY_REGISTRY['sobol'] is SobolStrategy
    reversed_keys = dict(reversed(list(params.items())))
    other = SobolStrategy(ParamDomain(reversed_keys), seed=SEED)
    assert list(MSQ(other, other.domain, n_permutations=32)) == actual
    with pytest.raises(RuntimeError, match='novel'):
        next(strategy)
    assert strategy.get_state()['raw_index'] == 1032


def test_sobol_msq_dedup_injection_filter_and_domain_changes(recorded_bars):
    params = _recorded_domain(recorded_bars)
    first = _project(_engine(params).random(1)[0], params)
    strategy = SobolStrategy(ParamDomain(params), seed=SEED)
    msq = MSQ(strategy, strategy.domain, n_permutations=8)
    msq.inject(first)
    injected = next(msq)
    assert injected['_injected'] is True and injected['_generation_index'] is None
    assert injected['_id'] in strategy._seen
    blocked = params['volume'][0]
    msq.set_filter('volume', lambda combo: combo['volume'] == blocked,
                   filter_type='exclude_value', filter_params={'param': 'volume', 'value': blocked})
    accepted = [next(msq) for _ in range(3)]
    assert all(combo['volume'] != blocked and combo['_id'] != injected['_id'] for combo in accepted)
    assert all(combo['_injected'] is False for combo in accepted)
    assert [combo['_round_index'] for combo in accepted] == [1, 2, 3]
    before = strategy.get_state()
    seen = set(strategy._seen)
    msq.remove_is('close', params['close'][-1])
    assert strategy.get_state()['raw_index'] == before['raw_index']
    assert strategy.generated_count == before['generated_count'] and strategy._seen == seen
    state = json.loads(json.dumps(msq.get_state()))
    restored_domain = ParamDomain(strategy.domain.params)
    restored_strategy = SobolStrategy(restored_domain, seed=SEED)
    restored = MSQ(restored_strategy, restored_domain)
    restored.set_state(state)
    restored_strategy.rebuild_seen_from_log(seen)
    expected = list(msq)
    assert list(restored) == expected
    assert all(combo['close'] != params['close'][-1] and combo['volume'] != blocked for combo in expected)
    assert len({combo['_id'] for combo in [injected, *accepted, *expected]}) == 8
    assert restored_strategy.get_state() == strategy.get_state()


@pytest.mark.parametrize('fault', ('version', 'seed', 'raw_index', 'generated_count',
                                  'domain_digest', 'scipy_version', 'unknown', 'missing'))
def test_sobol_checkpoint_json_roundtrip_and_rejection_is_atomic(recorded_bars, fault):
    params = _recorded_domain(recorded_bars)
    strategy = SobolStrategy(ParamDomain(params), seed=SEED)
    yielded = list(MSQ(strategy, strategy.domain, n_permutations=5))
    saved = json.loads(json.dumps(strategy.get_state()))
    assert saved['version'] == 1 and saved['seed'] == SEED
    peer = SobolStrategy(ParamDomain(params), seed=SEED)
    peer.set_state(saved)
    peer.rebuild_seen_from_log(combo['_id'] for combo in yielded)
    broken = copy.deepcopy(saved)
    if fault == 'version':
        broken['version'] = 2
    elif fault == 'seed':
        broken['seed'] += 1
    elif fault == 'raw_index':
        broken['raw_index'] = -1
    elif fault == 'generated_count':
        broken['generated_count'] = saved['raw_index'] + 1
    elif fault in ('domain_digest', 'scipy_version'):
        broken[fault] = 'mismatch'
    elif fault == 'unknown':
        broken['scramble'] = True
    else:
        del broken['version']
    seen = set(strategy._seen)
    with pytest.raises(ValueError, match=r'checkpoint|state|version|seed|domain|SciPy|counters'):
        strategy.set_state(broken)
    assert strategy.get_state() == saved and strategy._seen == seen
    assert [next(strategy) for _ in range(5)] == [next(peer) for _ in range(5)]


@pytest.mark.parametrize('declaration', (
    None, {'seed': None}, {'seed': True}, {'seed': -1}, {'seed': 1.5},
    {'seed': '42'}, {'seed': '{fee}'}, {'seed': []},
    {'seed': SEED, 'scramble': True}, {'seed': SEED, 'bits': 30}, {'seed': SEED, 'unknown': 1},
))
def test_sobol_yaml_native_equivalence_and_invalid_configuration(declaration):
    config = _config()
    before = copy.deepcopy(config)
    compiled = build_search_strategy(config)
    native = SobolStrategy(ParamDomain(config['sfd']['params']), seed=SEED)
    assert validate(config).valid and type(compiled) is SobolStrategy
    assert [next(compiled) for _ in range(6)] == [next(native) for _ in range(6)]
    assert compiled.get_state() == native.get_state() and config == before
    config['uel']['search_strategy'] = {'type': 'sobol', **(declaration or {})}
    result = validate(config)
    assert not result.valid
    assert any(error.path.startswith('uel.search_strategy') for error in result.errors)
    with pytest.raises(ValueError, match=r'seed|scramble|bits|unknown|[Uu]nknown'):
        build_search_strategy(config)
    if declaration is not None and set(declaration) == {'seed'}:
        with pytest.raises(ValueError, match='seed'):
            SobolStrategy(ParamDomain(before['sfd']['params']), seed=declaration['seed'])
    with pytest.raises(ValueError, match='parameter'):
        SobolStrategy(ParamDomain({}), seed=SEED)


def test_sobol_recorded_run_resume_preserves_ordered_results_and_predictions(recorded_bars, partial_run, tmp_path):
    config, partial_path = partial_run
    full = _loop(copy.deepcopy(config), recorded_bars, tmp_path / 'full', native=True)
    _run(full)
    path = tmp_path / 'resumed'
    shutil.copytree(partial_path, path)
    resumed = _loop(copy.deepcopy(config), recorded_bars, path)
    _run(resumed, resume=True)
    full_rows, full_records, full_audit = _science(tmp_path / 'full')
    rows, records, audit = _science(path)
    assert rows.equals(full_rows) and records == full_records and audit == full_audit
    assert rows.height == len(records) == 6
    assert rows['_round_index'].to_list() == [record['_round_index'] for record in records] == list(range(6))
    assert len(set(rows['_id'].to_list())) == 6
    assert [record['round_id'] for record in records] == rows['_id'].to_list()
    assert all(record['probs'] and record['preds'] and record['execution']['pos'] for record in records)
    assert rows['C'].to_list()[2:] == [1.0] * 4
    assert any(item['operation'] == 'remove_is' for item in audit[0]['interventions'])
    assert resumed._search_strategy.domain.params['C'] == [1.0]
    assert resumed._search_strategy.get_state() == full._search_strategy.get_state()
    checkpoint = json.loads((path / 'checkpoint.json').read_text())
    assert checkpoint['domain_state']['C'] == [1.0]
    assert checkpoint['msq_state']['strategy_state']['raw_index'] >= 6


@pytest.mark.parametrize('fault', ('seed', 'version', 'raw_index', 'scipy_version', 'domain_digest'))
def test_sobol_resume_rejection_preserves_artifacts(recorded_bars, partial_run, tmp_path, fault):
    config, partial_path = partial_run
    path = tmp_path / 'rejected'
    shutil.copytree(partial_path, path)
    changed = copy.deepcopy(config)
    checkpoint_path = path / 'checkpoint.json'
    checkpoint = json.loads(checkpoint_path.read_text())
    if fault == 'seed':
        changed['uel']['search_strategy']['seed'] += 1
    else:
        checkpoint['msq_state']['strategy_state'][fault] = -1 if fault in ('version', 'raw_index') else 'mismatch'
        checkpoint_path.write_text(json.dumps(checkpoint))
    before = _snapshot(path)
    loop = _loop(changed, recorded_bars, path)
    with pytest.raises(ValueError, match=r'[Cc]ontent hash|checkpoint|version|seed|domain|SciPy|counters'):
        _run(loop, resume=True)
    assert _snapshot(path) == before


def test_existing_random_grid_and_factorize_admission_remain_unchanged(recorded_bars, tmp_path):
    params = _recorded_domain(recorded_bars)
    rng = random.Random(SEED)
    random_strategy = RandomStrategy(ParamDomain(params), seed=SEED)
    expected = [{key: rng.choice(params[key]) for key in sorted(params)} for _ in range(6)]
    assert [_clean(next(random_strategy)) for _ in range(6)] == expected
    grid = list(GridStrategy(ParamDomain(params)))
    keys = sorted(params)
    expected_grid = [dict(zip(keys, values, strict=True))
                     for values in (reversed(combo) for combo in itertools.product(*(params[key] for key in reversed(keys))))]
    assert [_clean(combo) for combo in grid] == expected_grid
    for kind, cls in (('random', RandomStrategy), ('grid', GridStrategy)):
        config = _config()
        config['uel']['search_strategy'] = {'type': kind, 'seed': SEED}
        compiled = build_search_strategy(config)
        native = cls(ParamDomain(config['sfd']['params']), seed=SEED)
        assert compiled.get_state() == native.get_state()
        assert [next(compiled) for _ in range(6)] == [next(native) for _ in range(6)]
    loop = _loop(_config(), recorded_bars, tmp_path)
    with pytest.raises(ValueError, match='factorize requires'):
        _run(loop, factorize=True)
    assert _snapshot(tmp_path) == {}
