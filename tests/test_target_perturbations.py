import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
import pytest

from limen._parameter_identity import execution_hash
from limen.experiment import UniversalExperimentLoop
from limen.targets import ForwardBreakoutTarget, TripleBarrierTarget
from limen.inference import Trainer
from limen.yaml import CompiledSFD, build_search_strategy, validate
from tests.test_factorized_preparation import _bars, _config


@pytest.fixture(scope='module')
def market():
    return pl.read_parquet(Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet', n_rows=512)


def test_forward_noise_is_a_replayable_candidate_level_transformation(market):
    identity = execution_hash({'horizon_jitter': 3, 'noise_seed': 7, 'threshold': .001})
    target = ForwardBreakoutTarget(market, 'target')
    observed = target.transform(market, threshold=.001, horizon_jitter=3,
                                threshold_noise_std=.05, noise_seed=7, permutation_id=identity)
    realized = dict(target.perturbation_parameters)
    reference = ForwardBreakoutTarget(market, 'target').transform(market,
        forward_periods=realized['forward_periods'], threshold=realized['threshold'])
    assert_frame_equal(observed, reference)
    assert 21 <= realized['forward_periods'] <= 27 and realized['threshold'] != .001
    assert_frame_equal(observed, target.transform(market, threshold=.001, horizon_jitter=3,
        threshold_noise_std=.05, noise_seed=7, permutation_id=identity))
    target.transform(market, threshold=.001, horizon_jitter=3, threshold_noise_std=.05,
                      noise_seed=7, permutation_id=execution_hash({'noise_seed': 8}))
    assert target.perturbation_parameters['threshold'] != realized['threshold']


def test_barrier_noise_matches_realized_first_touch_and_training_smoothing(market):
    target = TripleBarrierTarget(market, 'target', max_horizon=10, span=10, min_periods=10)
    observed = target.transform(market, horizon_jitter=3, barrier_noise_std=.05,
                                noise_seed=7, permutation_id=execution_hash({'noise_seed': 7}))
    settings = {key: value for key, value in target.perturbation_parameters.items() if key != 'permutation_id'}
    reference = TripleBarrierTarget(market, 'target', span=10, min_periods=10, **settings).transform(market)
    assert_frame_equal(observed, reference)
    assert target.upper_multiple == target.lower_multiple == 1 and target.max_horizon == 10
    binary = ForwardBreakoutTarget(market, 'binary')
    hard = binary.transform(market, threshold=.001)
    train = binary.transform(market, threshold=.001, label_smoothing=.2, training=True)
    evaluation = binary.transform(market, threshold=.001, label_smoothing=.2, training=False)
    assert_frame_equal(hard, evaluation)
    np.testing.assert_allclose(train['binary'].drop_nulls(), hard['binary'].drop_nulls() * .8 + .1)


def config():
    result = _config()
    manifest = result['sfd']['manifest']
    manifest['reference_architecture'] = 'limen.sfd.reference_architecture.dlinear_regressor'
    controls = {'forward_periods': [3], 'threshold': [.001], 'horizon_jitter': [1],
                'threshold_noise_std': [.05], 'noise_seed': [7, 9], 'label_smoothing': [.2]}
    manifest['target'] = {'name': 'up_next', 'class': 'limen.targets.ForwardBreakoutTarget',
                           'transform_params': {key: '{' + key + '}' for key in controls}}
    result['sfd']['params'] = {'lookback': [2], 'fee': [4.], **controls}
    result['uel']['n_permutations'] = 2
    return result


@pytest.mark.parametrize('workers', [1, 2])
def test_native_manifest_binds_noise_to_logged_candidate_ids(tmp_path, workers):
    reference = config()
    assert validate(reference).valid
    experiment = UniversalExperimentLoop(sfd=CompiledSFD(reference), data=_bars(), yaml_reference=reference,
        search_strategy=build_search_strategy(reference), experiment_dir=tmp_path)
    experiment.run('noise', n_permutations=2, prep_each_round=True, progress_bar=False, post_processing=workers == 1,
                   context_params={'study': 'recorded-context-control'}, workers=workers, record_execution=True, record_model_outputs=True,
                   recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True))
    assert experiment.experiment_log.height == 2
    identities = experiment.experiment_log['id'].to_list()
    sensors = Trainer(tmp_path).train(identities)
    assert len(sensors) == 2
    for sensor, identity in zip(sensors, identities, strict=True):
        assert sensor._fitted_params['_target_cls_up_next'].perturbation_parameters['permutation_id'] == identity
        assert '_permutation_id' not in sensor.round_params
    for index, sensor in enumerate(sensors):
        params = dict(sensor.round_params)
        from limen.experiment._target_identity import bind
        with bind(experiment.manifest, params, experiment.experiment_log['id'][index]):
            prepared = experiment.manifest.prepare_data(experiment.data, params)
        assert '_permutation_id' not in params
        target = prepared['_fitted_params']['_target_cls_up_next']
        assert target.perturbation_parameters['permutation_id'] == experiment.experiment_log['id'][index]
        assert set(prepared['y_train'].to_numpy().ravel()) <= {.1, .9}
        assert set(prepared['y_val'].to_numpy().ravel()) <= {0, 1}
        assert set(prepared['y_test'].to_numpy().ravel()) <= {0, 1}
    with pytest.raises(ValueError, match='framework operand'):
        experiment.manifest.target_class_config.transform_params['permutation_id'] = 'override'
        experiment.manifest.prepare_data(experiment.data, sensors[0].round_params)


def test_real_cli_runs_the_declared_noise_sweep(tmp_path):
    from ruamel.yaml import YAML
    reference = config()
    reference['uel']['output_path'] = 'noise'
    path = tmp_path / 'noise.yaml'
    with path.open('w') as stream:
        YAML().dump(reference, stream)
    completed = subprocess.run([sys.executable, '-m', 'limen', 'run', '--no-progress-bar', str(path)],
        cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1])),
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    log = pl.read_csv(tmp_path / 'results/dev/noise/results.csv')
    assert log.height == 2 and log['noise_seed'].to_list() == [7, 9]
    assert log['strict_mode_error'].null_count() == 2


def test_factorized_noise_is_refused_before_artifacts(tmp_path):
    reference = config()
    directory = tmp_path / 'factorized'
    experiment = UniversalExperimentLoop(sfd=CompiledSFD(reference), data=_bars(), yaml_reference=reference,
        search_strategy=build_search_strategy(reference), experiment_dir=directory)
    with pytest.raises(ValueError, match='permutation-seeded target noise'):
        experiment.run('noise', n_permutations=2, prep_each_round=True, factorize=True, progress_bar=False)
    assert not directory.exists()


@pytest.mark.parametrize('controls', [
    {'horizon_jitter': True}, {'horizon_jitter': 24}, {'noise_seed': -1}, {'noise_seed': True},
    {'threshold_noise_std': -1}, {'threshold_noise_std': float('nan')},
    {'label_smoothing': 1}, {'label_smoothing': float('nan')},
])
def test_invalid_target_noise_controls_refuse(market, controls):
    with pytest.raises(ValueError):
        ForwardBreakoutTarget(market, 'target').transform(market, permutation_id='recorded-control', **controls)


def test_yaml_anchored_noise_controls(market):
    from ruamel.yaml import YAML
    settings = YAML().load('horizon_jitter: &jitter 1\nnoise_seed: &seed 7')
    target = ForwardBreakoutTarget(market, 'target')
    assert_frame_equal(target.transform(market, permutation_id='recorded-control', **settings),
                       target.transform(market, permutation_id='recorded-control', horizon_jitter=1, noise_seed=7))
    settings['noise_seed'] = YAML().load('value: &flag true')['value']
    with pytest.raises(ValueError, match='noise_seed'):
        target.transform(market, permutation_id='recorded-control', **settings)


def test_walk_forward_reconstruction_replays_the_candidate_noise(tmp_path):
    reference = config()
    manifest = reference['sfd']['manifest']
    manifest.pop('split_dates')
    manifest['split_walk_forward'] = {'n_folds': 2, 'test_bars': 30, 'purge_bars': 3,
                                     'embargo_bars': 1, 'anchored': True}
    manifest['reference_architecture'] = 'limen.sfd.reference_architecture.ridge_regressor'
    reference['sfd']['params'].update(alpha=[1.], solver=['svd'])
    experiment = UniversalExperimentLoop(sfd=CompiledSFD(reference), data=_bars(), yaml_reference=reference,
        search_strategy=build_search_strategy(reference), experiment_dir=tmp_path)
    experiment.run('fold_noise', n_permutations=2, prep_each_round=True, progress_bar=False, post_processing=True,
                   record_execution=True, record_model_outputs=True,
                   recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True))
    identities = experiment.experiment_log['id'].to_list()
    sensors = Trainer(tmp_path, data=_bars()).train(identities, fold=1)
    assert len(sensors) == 2
    for sensor, identity in zip(sensors, identities, strict=True):
        assert sensor._fitted_params['_target_cls_up_next'].perturbation_parameters['permutation_id'] == identity
        assert '_permutation_id' not in sensor.round_params
