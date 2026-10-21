import json

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from limen.experiment import UniversalExperimentLoop
from limen.experiment._parallel_worker import validate_parallel
from limen.yaml import CompiledSFD, build_search_strategy, parse, validate
from limen.yaml.compiler import build_manifest
from tests.test_parallel_rounds import ROOT, _TRANSIENT, _assert_same, recorded_source, records
from tests.test_record_model_outputs import _config

TEMPLATES = ('logreg_binary', 'lightgbm_binary', 'dlinear_regressor', 'ridge_regressor', 'xgboost_regressor',
             'rule_based', 'common_strategies', 'dollar_bar_crash_reversal')
SPLITS = {'train_start': '2024-01-01', 'train_end': '2024-01-23', 'val_start': '2024-01-24', 'val_end': '2024-01-30',
          'test_start': '2024-01-31', 'test_end': '2024-02-06'}


def template(name, **overrides):
    config, errors = parse((ROOT / f'limen/yaml/templates/{name}.yaml').read_text())
    assert not errors
    domain = config['sfd']['params']
    for key, value in (('n_jobs', 1), ('device', 'cpu')):
        if key in domain:
            domain[key] = [value]
    domain.update(overrides)
    result = validate(config)
    assert result.valid, result.errors
    return config


def admit(config, context=None):
    validate_parallel(build_manifest(config), config, build_search_strategy(config).domain.params, context)


def dlinear_config():
    config = _config('ridge')
    config['sfd']['manifest']['reference_architecture'] = 'limen.sfd.reference_architecture.dlinear_regressor'
    config['sfd']['params'] = {'kernel_size': [3], 'alpha': [1.0, 10.0]}
    return config


def rule_based_config():
    config, errors = parse((ROOT / 'limen/yaml/templates/rule_based.yaml').read_text())
    assert not errors
    config['sfd']['manifest']['split_dates'] = dict(SPLITS)
    config['sfd']['params'] = {'rsi_period': [7, 14], 'rsi_threshold': [30], 'ema_period': [50]}
    config['uel'].update(n_permutations=2, search_strategy={'type': 'grid'}, checkpoint_interval=1)
    return config


FAMILIES = {'dlinear': dlinear_config, 'ridge': lambda: _config('ridge'),
            'xgboost': lambda: _config('xgboost'), 'rule_based': rule_based_config}


def run(config, directory, *, workers, resume=False):
    loop = UniversalExperimentLoop(sfd=CompiledSFD(config), data=recorded_source(), experiment_dir=directory,
                                   yaml_reference=config, search_strategy=build_search_strategy(config), checkpoint_interval=1)
    loop.run('admitted_cpu_rounds', n_permutations=2, prep_each_round=True, progress_bar=False,
             record_execution=True, record_model_outputs=True, workers=workers, resume=resume,
             recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True))
    return loop


@pytest.mark.parametrize('name', TEMPLATES)
def test_shipped_templates_admit_cpu_rounds(name):
    admit(template(name))


def test_exposed_knobs_follow_backend_defaults():
    admit(template('xgboost_regressor', device=[None]))
    with pytest.raises(ValueError, match='CPU backend'):
        admit(template('xgboost_regressor'), {'device': 'cuda'})
    with pytest.raises(ValueError, match='n_jobs=1'):
        admit(template('xgboost_regressor', n_jobs=[None]))


@pytest.mark.parametrize('family', FAMILIES)
def test_compatible_families_match_serial_and_resume(tmp_path, family):
    config = FAMILIES[family]()
    serial, parallel = tmp_path / 'serial', tmp_path / 'parallel'
    run(config, serial, workers=1)
    run(config, parallel, workers=2)
    left, right = pl.read_csv(serial / 'results.csv'), pl.read_csv(parallel / 'results.csv')
    columns = [name for name in left.columns if name not in _TRANSIENT]
    assert_frame_equal(left.select(columns), right.select(columns), check_exact=False, rel_tol=1e-10, abs_tol=1e-12)
    saved = records(parallel)
    _assert_same(saved, records(serial))
    assert len(saved) == 2
    for row in saved:
        assert row['probs'] is None
        assert all(value in (0, 1) for value in row['preds']) is (family == 'rule_based')
    assert json.loads((parallel / 'metadata.json').read_text())['workers'] == 2
    assert run(config, parallel, workers=2, resume=True).experiment_log.height == 2
