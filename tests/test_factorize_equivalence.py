"""Opt-in factorized grids must preserve the original scientific result surface."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import sys
import warnings
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from limen.experiment import UniversalExperimentLoop
from limen.experiment._factorize import FactorizedRounds
from limen.experiment.errors import StrictModeError
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search.grid_strategy import GridStrategy
from limen.experiment.param_search.random_strategy import RandomStrategy
from limen.yaml import validate
from limen.yaml.compiler import build_manifest

from tests.fixtures.factorize.generate import (
    BASELINE, CASES, MARKET, canonical, component_digests, digest, execute, manifest,
)

GOLDENS = Path(__file__).parent / 'fixtures/factorize'


def _loop(case: str, directory: Path, *, settings: dict | None = None, **kwargs: object) -> UniversalExperimentLoop:
    cfg = manifest(case)
    if settings:
        cfg['sfd']['params'].update(settings)
    params = cfg['sfd']['params']
    compiled = build_manifest(cfg)
    sfd = SimpleNamespace(__name__='limen.sfd.factorize_fixture',
                          params=lambda: params, manifest=lambda: compiled)
    strategy = kwargs.pop('strategy', GridStrategy(ParamDomain(params)))
    return UniversalExperimentLoop(sfd=sfd, data=pl.read_parquet(MARKET),
                                   search_strategy=strategy, experiment_dir=directory,
                                   **kwargs)


def _numeric_equal(a: object, b: object) -> None:
    if type(a) is float:
        assert type(b) is float
        if math.isnan(a):
            assert math.isnan(b)
        else:
            assert a == b
            if a == 0:
                assert math.copysign(1., a) == math.copysign(1., b)
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _numeric_equal(x, y)
    elif isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b)
        for key in a:
            _numeric_equal(a[key], b[key])
    else:
        assert type(a) is type(b) and a == b


def test_cli_manifest_switch(tmp_path: Path) -> None:
    cfg = manifest('binary_costs')
    for value in (True, False):
        cfg['uel']['factorize'] = value
        assert validate(cfg).valid
    for value in (0, 1, 'true', None, []):
        cfg['uel']['factorize'] = value
        assert not validate(cfg).valid, value
    loop = _loop('directional_barriers', tmp_path / 'direct')
    for value in (0, 1, 'true', None):
        with pytest.raises(ValueError, match='factorize must be a bool'):
            loop.run('invalid', n_permutations=1, factorize=value, progress_bar=False)
    assert not (tmp_path / 'direct/results.csv').exists()


def test_four_cli_experiments_match_pre_feature_baseline_at_12_decimals(tmp_path: Path) -> None:
    for name in CASES:
        baseline = json.loads((GOLDENS / f'{name}.json').read_text())
        assert baseline['source_sha'] == BASELINE
        assert baseline['market_sha256'] == hashlib.sha256(MARKET.read_bytes()).hexdigest()
        off = execute(name, tmp_path / name / 'off')
        on = execute(name, tmp_path / name / 'on', factorize=True)
        # Frozen goldens require the locked Linux/Python 3.10 numerical runtime.
        if sys.platform == 'linux' and sys.version_info[:2] == (3, 10) and np.__version__ == '2.2.6':
            actual = component_digests(off)
            differences = {key: (expected, actual.get(key))
                           for key, expected in baseline['component_digests'].items()
                           if actual.get(key) != expected}
            assert digest(off) == baseline['digest_12dp'], f'{name}: original Limen vs switch-off: {differences}'
            assert digest(on) == baseline['digest_12dp'], f'{name}: original Limen vs factorized'
        else:
            assert digest(off) == digest(on), f'{name}: current on/off parity'
        assert off['columns'] == on['columns'] == baseline['columns']
        assert len(off['rows']) == len(on['rows']) == baseline['row_count']
        assert [r['round_id'] for r in on['round_data']] == baseline['round_ids']
        assert [{k:v for k,v in row.items() if k != 'execution_time'} for row in off['rows']] == [
            {k:v for k,v in row.items() if k != 'execution_time'} for row in on['rows']]
        _numeric_equal(off['round_data'], on['round_data'])
        assert on['metadata']['factorize'] is True
        assert 'factorize' not in off['metadata']
        assert off['metadata']['manifest_id'] != on['metadata']['manifest_id']

        rows = off['rows']
        col = 'pnl_per_bar_bps_test' if name == 'rule_based' else 'backtest_pnl_per_bar_bps'
        assert len({row[col] for row in rows}) > 1, name
        assert any(any(x != 0 for x in record['preds']) for record in off['round_data'])
        if name == 'interleaved':
            assert len({tuple(record['preds']) for record in off['round_data']}) > 1
        if name == 'binary_costs':
            assert off['round_data'][0]['probs']
            assert off['round_data'][0]['execution']['pos']
        if name == 'directional_barriers':
            assert any(row['tp'] == '' for row in rows)
            assert any(row['sl'] == '' for row in rows)
            assert len({row[col] for row in rows if row['tp']}) > 1

    altered = copy.deepcopy(off)
    altered['rows'][0][col] = '123.123456789124'
    assert digest(altered) != digest(off)
    assert canonical(1.000000000001) != canonical(1.000000000002)
    assert canonical(-0.0) != canonical(0.0)
    assert canonical(float('nan')) != canonical(None)
    assert canonical(float('inf')) != canonical(float('-inf'))


def test_one_signal_generation_per_unique_signal_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, expected in (('binary_costs', 1), ('directional_barriers', 1),
                           ('interleaved', 4), ('rule_based', 1)):
        loop = _loop(name, tmp_path / name)
        counts = {'prep': 0, 'model': 0}
        orig_prep, orig_model = type(loop.manifest).prepare_data, type(loop.manifest).run_model
        assert orig_prep and orig_model

        def prep(*args: object, _counts: dict = counts, _fn: object = orig_prep,
                 **kwargs: object) -> dict:
            _counts['prep'] += 1
            return _fn(*args, **kwargs)

        def model(*args: object, _counts: dict = counts, _fn: object = orig_model,
                  **kwargs: object) -> dict:
            _counts['model'] += 1
            return _fn(*args, **kwargs)

        with monkeypatch.context() as instrumentation:
            instrumentation.setattr(type(loop.manifest), 'prepare_data', prep)
            instrumentation.setattr(type(loop.manifest), 'run_model', model)
            loop.run(name, n_permutations=manifest(name)['uel']['n_permutations'],
                     prep_each_round=True, factorize=True, progress_bar=False)
        assert counts == {'prep': expected, 'model': expected}, (name, counts)


def test_artifacts_and_replay_identical(tmp_path: Path) -> None:
    left, right = [], []
    for enabled in (False, True):
        loop = _loop('binary_costs', tmp_path / str(enabled))
        loop.run('binary_costs', n_permutations=8, prep_each_round=True,
                 factorize=enabled, progress_bar=False, post_processing=True,
                 record_execution=True, record_model_outputs=True)
        (right if enabled else left).append(loop)
    baseline, changed = left[0], right[0]
    _numeric_equal(baseline.experiment_log.drop('execution_time').to_dicts(),
                   changed.experiment_log.drop('execution_time').to_dicts())
    for attribute in ('experiment_confusion_metrics', 'experiment_backtest_results'):
        old, new = getattr(baseline, attribute), getattr(changed, attribute)
        _numeric_equal(old.to_dict('records'), new.to_dict('records'))
    for i in range(8):
        np.testing.assert_array_equal(baseline.preds[i], changed.preds[i])
        _numeric_equal(baseline._alignment[i]['execution'], changed._alignment[i]['execution'])
        _numeric_equal(baseline._alignment[i]['market'], changed._alignment[i]['market'])
        _numeric_equal(baseline._alignment[i]['model_outputs'], changed._alignment[i]['model_outputs'])


@pytest.mark.parametrize('case', [
    'no_axes', 'model_overlap', 'feature_braced', 'feature_bare', 'feature_template',
    'feature_format', 'feature_conversion', 'feature_access', 'feature_nested_format',
    'random', 'pruning', 'callback', 'unsafe_lightgbm_later', 'scaler',
    'calibration', 'feature_groups', 'pre_split', 'event', 'unknown_arch',
    'no_strategy', 'walk_forward', 'selector_group', 'source_reference', 'unsafe_seed_later', 'event_axis',
])
def test_unsafe_and_nondeterministic_paths_fail_before_writes(
    case: str, tmp_path: Path,
) -> None:
    cfg = manifest('directional_barriers')
    m = cfg['sfd']['manifest']
    p = cfg['sfd']['params']
    extra = {}
    if case == 'no_axes':
        m['backtest'] = {'fee_bps': 5.}
    elif case == 'model_overlap':
        m['backtest']['notional_rate'] = '{alpha}'
        p['alpha'] = [0.5, 1.]
    elif case.startswith('feature_'):
        m['features'][0]['params']['end'] = {
            'feature_braced': '{fee}', 'feature_bare': 'fee', 'feature_template': 'col_{fee}',
            'feature_format': '{fee:.1f}', 'feature_conversion': '{fee!s}',
            'feature_access': '{fee.real}', 'feature_nested_format': '{alpha:{fee}}',
        }.get(case, '{lookback_end}')
        if case == 'feature_groups':
            m['backtest']['fee_bps'] = '{feature_groups}'
            p['feature_groups'] = [5., 10.]
    elif case == 'random':
        extra['strategy'] = RandomStrategy(ParamDomain(p), seed=42)
    elif case == 'pruning':
        extra['pruning_strategies'] = [object()]
    elif case == 'callback':
        extra['intra_callback'] = lambda *_args: None
    elif case == 'unsafe_lightgbm_later':
        cfg = manifest('binary_costs')
        cfg['sfd']['params']['deterministic'] = [True, False]
    elif case == 'unsafe_seed_later':
        cfg = manifest('binary_costs')
        cfg['sfd']['params']['random_state'] = [42, None]
    elif case == 'scaler':
        m['scaler'] = {'class': 'limen.scalers.RobustScaler'}
    elif case == 'calibration':
        m['calibration'] = {'threshold_function': {
            'func': 'limen.calibration.grid_threshold_optimizer',
            'params': {'threshold_min': 0.2, 'threshold_max': 0.8, 'threshold_step': 0.1}}}
    elif case == 'feature_groups':
        m['backtest']['fee_bps'] = '{feature_groups}'
        p['feature_groups'] = [5., 10.]
    elif case == 'pre_split':
        pass
    elif case == 'event':
        m['backtest']['prediction_mode'] = 'target_exposure'
    elif case == 'event_axis':
        p['fee'] = [1., 2.]
        m['backtest']['max_exposure'] = '{fee}'
    elif case == 'unknown_arch':
        m['reference_architecture'] = 'limen.sfd.reference_architecture.ridge_regressor.ridge_regressor'
    elif case == 'no_strategy':
        extra['strategy'] = None
    elif case == 'walk_forward':
        pass
    elif case == 'selector_group':
        m['features'][0]['include_if'] = 'use_custom'
    elif case == 'source_reference':
        m['data_source']['params']['kline_size'] = '{fee}'
    params = cfg['sfd']['params']
    built = build_manifest(cfg)
    if case == 'pre_split':
        built.pre_split_data_selector = (lambda data: data, {})
    if case == 'walk_forward':
        built.split_walk_forward = object()
    sfd = SimpleNamespace(__name__='limen.sfd.fixture', params=lambda: params, manifest=lambda: built)
    loop = UniversalExperimentLoop(sfd=sfd, data=pl.read_parquet(MARKET),
                                   search_strategy=extra.pop('strategy', GridStrategy(ParamDomain(params))),
                                   experiment_dir=tmp_path / case, **extra)
    with pytest.raises((ValueError, TypeError)):
        loop.run(case, n_permutations=2, prep_each_round=True, factorize=True, progress_bar=False)
    for file in ('metadata.json', 'checkpoint.json', 'results.csv', 'round_data.jsonl'):
        assert not (tmp_path / case / file).exists()


@pytest.mark.parametrize('context', [{'deterministic': False}, {'n_jobs': 8}, {'random_state': None}, {'fee': 9.}])
def test_context_overrides_fail_before_writes(tmp_path: Path, context: dict) -> None:
    loop = _loop('binary_costs', tmp_path)
    with pytest.raises(ValueError, match='factorize requires'):
        loop.run('context', n_permutations=8, context_params=context,
                 prep_each_round=True, factorize=True, progress_bar=False)
    assert not (tmp_path / 'metadata.json').exists()


@pytest.mark.parametrize('op', [{'op': 'inject', 'combo': {'deterministic': False}},
                              {'op': 'inject_value', 'param': 'n_jobs', 'value': 8}])
def test_interventions_fail_before_writes(tmp_path: Path, op: dict) -> None:
    (tmp_path / 'interventions.json').write_text(json.dumps([op]))
    loop = _loop('binary_costs', tmp_path)
    with pytest.raises(ValueError, match='interventions'):
        loop.run('interventions', n_permutations=8, prep_each_round=True,
                 factorize=True, progress_bar=False)
    assert not (tmp_path / 'metadata.json').exists()


def test_cache_releases_fitted_model_and_rejects_new_candidates(tmp_path: Path) -> None:
    loop = _loop('binary_costs', tmp_path)
    domain = manifest('binary_costs')['sfd']['params']
    rounds = FactorizedRounds(manifest=loop.manifest, strategy=GridStrategy(ParamDomain(domain)),
                              domain=domain, pruning=False, callback=False, context=None,
                              prep=loop.manifest.prepare_data, model=loop.manifest.run_model, data=loop.data,
                              record_execution=True, record_model_outputs=True)
    params = {key: values[0] for key, values in domain.items()}
    data, result, _ = rounds.evaluate(params)
    fitted = weakref.ref(result['_model'])
    del data, result
    assert fitted() is None
    domain['deterministic'].append(False)
    with pytest.raises(ValueError, match='outside the preflight domain'):
        rounds.evaluate({**params, 'deterministic': False})
    with pytest.raises(ValueError, match='outside the preflight domain'):
        rounds.evaluate({**params, 'deterministic': 1})


@pytest.mark.parametrize('override', ['prep', 'model', 'manifest_prepare', 'manifest_model'])
def test_overridden_pipeline_fails_before_writes(tmp_path: Path, override: str) -> None:
    loop = _loop('directional_barriers', tmp_path)
    if override.startswith('manifest_'):
        owner, name = loop.manifest, {'manifest_prepare': 'prepare_data', 'manifest_model': 'run_model'}[override]
    else:
        owner, name = loop, override
    original = getattr(owner, name)

    def dependent(*args: object, **kwargs: object) -> dict:
        params = kwargs['round_params']
        return original(*args, **{**kwargs, 'round_params': {**params, 'alpha': params['fee']}})

    setattr(owner, name, dependent)
    with pytest.raises(ValueError, match='factorize requires'):
        loop.run('override', n_permutations=8, prep_each_round=True,
                 factorize=True, progress_bar=False)
    assert not (tmp_path / 'metadata.json').exists()


def test_resume_and_disabled_mode_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    full = _loop('rule_based', tmp_path / 'full', checkpoint_interval=1)
    full.run('resume', n_permutations=16, prep_each_round=True, factorize=True, progress_bar=False)
    split = _loop('rule_based', tmp_path / 'split', checkpoint_interval=1)
    original = split._checkpoint

    def stop(*args: object, **kwargs: object) -> None:
        original(*args, **kwargs)
        if args[4] == 3:
            split._shutdown_requested = True

    monkeypatch.setattr(split, '_checkpoint', stop)
    split.run('resume', n_permutations=16, prep_each_round=True, factorize=True, progress_bar=False)
    assert (tmp_path / 'split/checkpoint.json').exists()
    mismatch = _loop('rule_based', tmp_path / 'split', checkpoint_interval=1)
    files = {name: (tmp_path / 'split' / name).read_bytes() for name in
             ('checkpoint.json','metadata.json','results.csv','round_data.jsonl')}
    with pytest.raises(ValueError, match='factorize setting'):
        mismatch.run('resume', n_permutations=16, prep_each_round=True,
                     factorize=False, resume=True, progress_bar=False)
    assert files == {name: (tmp_path / 'split' / name).read_bytes() for name in files}
    resumed = _loop('rule_based', tmp_path / 'split', checkpoint_interval=1)
    resumed.run('resume', n_permutations=16, prep_each_round=True,
                factorize=True, resume=True, progress_bar=False)
    def parsed(path: Path) -> object:
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        return canonical([{k:v for k,v in row.items() if k != 'execution_time'}
                          for row in rows], in_csv=True)

    assert parsed(tmp_path / 'full/results.csv') == parsed(tmp_path / 'split/results.csv')

    legacy = _loop('rule_based', tmp_path / 'legacy', checkpoint_interval=1)
    legacy.run('legacy', n_permutations=16, prep_each_round=True,
               factorize=False, progress_bar=False)
    stored = {name: (tmp_path / 'legacy' / name).read_bytes() for name in
              ('metadata.json', 'checkpoint.json', 'results.csv', 'round_data.jsonl')}
    assert 'factorize' not in json.loads(stored['metadata.json'])
    changed = _loop('rule_based', tmp_path / 'legacy', checkpoint_interval=1)
    with pytest.raises(ValueError, match='factorize setting'):
        changed.run('legacy', n_permutations=16, prep_each_round=True,
                    factorize=True, resume=True, progress_bar=False)
    assert stored == {name: (tmp_path / 'legacy' / name).read_bytes() for name in stored}



def test_failed_round_warning_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = []
    for mode in (False, True):
        root = tmp_path / str(mode)
        loop = _loop('directional_barriers', root)

        def fail_prep(*_args: object, **_kwargs: object) -> dict:
            warnings.warn('signal-preparation-warning', RuntimeWarning, stacklevel=2)
            raise StrictModeError('synthetic strict failure')

        with monkeypatch.context() as instrumentation:
            instrumentation.setattr(type(loop.manifest), 'prepare_data', fail_prep)
            loop.run('fail', n_permutations=8, prep_each_round=True,
                     factorize=mode, progress_bar=False)
        with (root / 'results.csv').open(newline='') as stream:
            output.append(list(csv.DictReader(stream)))
    assert all(row['_warnings'] == '["signal-preparation-warning"]' for row in output[0])
    assert [{k:v for k,v in row.items() if k != 'execution_time'} for row in output[0]] == [
        {k:v for k,v in row.items() if k != 'execution_time'} for row in output[1]]
