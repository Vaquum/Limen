import copy
import json
from io import StringIO
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner
from sklearn.calibration import CalibratedClassifierCV
from threadpoolctl import threadpool_limits

from limen.experiment import MLManifest, UniversalExperimentLoop
from limen.cli.main import cli
from limen.experiment import _round_storage as _storage
from limen.experiment._factorized_admission import validate_candidate
from limen.experiment.errors import StrictModeError
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import GridStrategy
from limen.inference import Trainer
from limen.yaml import CompiledSFD, build_search_strategy, parse, validate
from limen.yaml.config import round_trip_yaml
from tests.test_column_shims import _config as _strict_config
from tests.test_column_shims import _native as _strict_native
from tests.test_sklearn_input_boundaries import _config, _direct_logreg, FIXTURE

_INVALID = ['false', 'no', '', 0, 1, 0.0, 1.0, None, [], {}, np.bool_(False)]


@pytest.fixture(scope='module')
def bars():
    return pl.read_parquet(FIXTURE, n_rows=336)


@pytest.fixture(autouse=True)
def single_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.mark.parametrize('name', ['strict_mode', 'use_calibration', 'use_threshold'])
@pytest.mark.parametrize('value', _INVALID)
@pytest.mark.parametrize('mode', ['development', 'production'])
def test_declared_controls_reject_non_booleans(name, value, mode):
    config = _config('logreg', 'sigmoid')
    config['metadata']['mode'] = mode
    if name == 'strict_mode':
        config['sfd']['manifest'][name] = value
    else:
        config['sfd']['params'][name] = [True, value]
    result = validate(config)
    assert not result.valid
    assert any(name in error.path and 'must be a bool' in error.message for error in result.errors)
    with pytest.raises(ValueError, match=rf'{name}.*must be a bool'):
        CompiledSFD(config)
    with pytest.raises(ValueError, match=rf'{name}.*must be a bool'):
        build_search_strategy(config)
    if name == 'strict_mode':
        with pytest.raises(ValueError, match='strict_mode must be a bool'):
            CompiledSFD(config).manifest()
        with pytest.raises(ValueError, match='strict_mode must be a bool'):
            MLManifest(strict_mode=value)
        with pytest.raises(ValueError, match='strict_mode must be a bool'):
            MLManifest().set_strict_mode(value)


def _run(loop, **kwargs):
    loop.run('boolean_controls', n_permutations=1, prep_each_round=True,
             random_search=False, progress_bar=False, **kwargs)


def _forbidden(*args, **kwargs):
    raise AssertionError('Invalid control reached preparation or fitting')


@pytest.mark.parametrize('msq', [False, True])
@pytest.mark.parametrize('name', ['use_calibration', 'use_threshold'])
@pytest.mark.parametrize('where', ['domain', 'context'])
def test_native_whole_domain_and_context_admission(bars, tmp_path, monkeypatch, msq, name, where):
    config = _config('logreg')  # Controls also apply without a manifest calibration block.
    compiled = CompiledSFD(config)
    manifest = compiled.manifest()
    params = compiled.params()
    params[name] = [True, 'false'] if where == 'domain' else [True]
    sfd = SimpleNamespace(params=lambda: copy.deepcopy(params), manifest=lambda: manifest)
    strategy = GridStrategy(ParamDomain(params)) if msq else None
    directory = tmp_path / 'refused'
    loop = UniversalExperimentLoop(sfd=sfd, data=bars, search_strategy=strategy, experiment_dir=directory)
    monkeypatch.setattr(manifest, 'prepare_data', _forbidden)
    with pytest.raises(ValueError, match=rf'{name} must be a bool'):
        _run(loop, context_params={name: 'false'} if where == 'context' else None)
    assert not directory.exists()


@pytest.mark.parametrize('name', ['strict_mode', 'use_calibration', 'use_threshold'])
@pytest.mark.parametrize('value', _INVALID)
def test_effective_controls_precede_direct_preparation_and_model(bars, monkeypatch, name, value):
    compiled = CompiledSFD(_config('logreg', 'sigmoid'))
    manifest, params = compiled.manifest(), {key: values[0] for key, values in compiled.params().items()}
    if name == 'strict_mode':
        manifest.strict_mode = value
    else:
        params[name] = value
        params['use_threshold' if name == 'use_calibration' else 'use_calibration'] = False
    manifest.pre_split_data_selector = (_forbidden, {})
    with pytest.raises(ValueError, match=rf'{name} must be a bool'):
        manifest.prepare_data(bars, params)
    with pytest.raises(ValueError, match=rf'{name} must be a bool'):
        manifest.run_model({}, params)
    with pytest.raises(ValueError, match=rf'{name} must be a bool'):
        validate_candidate(manifest, params, frozenset())
    manifest.objective = object()  # Objective admission must not bypass the type check.
    with pytest.raises(ValueError, match=rf'{name} must be a bool'):
        manifest.prepare_data(bars, params)


@pytest.mark.parametrize('mode', [None, (True, True), (True, False), (False, True), (False, False)])
def test_recorded_calibration_modes_and_defaults(bars, mode):
    config = _config('logreg', 'sigmoid')
    compiled = CompiledSFD(config)
    manifest = compiled.manifest()
    params = {name: values[0] for name, values in compiled.params().items()}
    controls = {} if mode is None else dict(zip(('use_calibration', 'use_threshold'), mode, strict=True))
    data = manifest.prepare_data(bars, params | controls)
    result = manifest.run_model(data, params | controls)
    model = result['_model']
    probabilities = model.predict({'x_test': data['x_test']})['_probs']
    calibration, threshold = (True, True) if mode is None else mode
    assert isinstance(model._fitted_calibrator, CalibratedClassifierCV) is calibration
    expected = {(True, True): (0.4, 32), (True, False): (0.5, 13),
                (False, True): (0.45, 2), (False, False): (None, 1)}
    selected, positives = expected[calibration, threshold]
    assert result['optimal_threshold'] == pytest.approx(selected) if selected is not None else result['optimal_threshold'] is None
    assert sum(result['_preds']) == positives
    assert (result['val_score'] is not None) is threshold
    if calibration or threshold:
        np.testing.assert_array_equal(result['_preds'], np.asarray(probabilities) >= (selected or 0.5))
    baseline = _direct_logreg(data, params, method='sigmoid', use_calibration=calibration,
        use_threshold=threshold, threshold_params=manifest.prediction_calibration_config.threshold_params)
    np.testing.assert_allclose(probabilities, baseline['_probs'], rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(result['_preds'], baseline['_preds'])
    assert result['val_score'] == baseline['val_score']


@pytest.mark.parametrize('strict', [None, False, True])
@pytest.mark.parametrize('native', [False, True])
def test_recorded_strict_null_behavior(bars, strict, native):
    config = _strict_config()
    if strict is not None:
        config['sfd']['manifest']['strict_mode'] = strict
    manifest = _strict_native() if native else CompiledSFD(config).manifest()
    if native and strict is not None:
        manifest.set_strict_mode(strict)
    fault = bars.with_columns(pl.when(pl.col('datetime') == bars['datetime'][180]).then(None).otherwise(pl.col('open')).alias('open'))
    if strict:
        with pytest.raises(StrictModeError, match='Unexpected nulls in val split'):
            manifest.prepare_data(fault, {})
    else:
        assert manifest.prepare_data(fault, {})['x_val'].height == 46


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('msq', [False, True])
@pytest.mark.parametrize('validated', [False, True])
def test_yaml_boolean_anchors_survive_traversal_metadata_and_replay(bars, tmp_path, monkeypatch, enabled, msq, validated):
    monkeypatch.chdir(tmp_path)
    config = _config('logreg', 'sigmoid')
    config['sfd']['manifest'].pop('strict_mode', None)
    stream = StringIO()
    round_trip_yaml().dump(config, stream)
    text = stream.getvalue().replace('  manifest:\n', f'  manifest:\n    strict_mode: &control {str(enabled).lower()}\n')
    text = text.replace('\n  params:\n', '\n  params:\n    use_calibration: [*control]\n    use_threshold: [*control]\n', 1)
    parsed, errors = parse(text)
    assert not errors
    if validated:
        assert validate(parsed).valid
    compiled = CompiledSFD(parsed)
    assert compiled.manifest().strict_mode is enabled
    assert compiled.params()['use_calibration'][0] is enabled
    assert compiled.params()['use_threshold'][0] is enabled
    directory = tmp_path / 'anchored'
    loop = UniversalExperimentLoop(sfd=compiled, data=bars, experiment_dir=directory,
        yaml_reference=parsed, search_strategy=build_search_strategy(parsed) if msq else None)
    _run(loop, post_processing=True, record_model_outputs=msq)
    assert loop.round_params[0]['use_calibration'] is enabled
    assert isinstance(loop.models[0]._fitted_calibrator, CalibratedClassifierCV) is enabled
    if not msq:
        return  # Legacy traversal has no persisted Trainer metadata contract.
    reference = json.loads((directory / 'metadata.json').read_text())['yaml_reference']
    assert reference['sfd']['manifest']['strict_mode'] is enabled
    assert reference['sfd']['params']['use_calibration'][0] is enabled
    row = json.loads(_storage.text(_storage.round_path(directory)))
    sensors = Trainer(directory, data=bars).train([row['round_id']])
    assert len(sensors) == 1
    assert isinstance(sensors[0]._model._fitted_calibrator, CalibratedClassifierCV) is enabled


@pytest.mark.parametrize('name', ['use_calibration', 'use_threshold'])
@pytest.mark.parametrize('method', ['inject', 'inject_value'])
def test_effective_msq_injections_are_rejected_before_another_fit(bars, tmp_path, monkeypatch, name, method):
    from limen.sfd.reference_architecture import LogRegBinary

    config = _config('logreg', 'sigmoid')
    config['sfd']['params'].update(use_calibration=[False], use_threshold=[False])
    config['sfd']['params']['C'] = [0.3, 1.0]
    compiled = CompiledSFD(config)
    calls, injected = [], []
    train = LogRegBinary.train

    def fit(self, *args, **kwargs):
        calls.append(True)
        return train(self, *args, **kwargs)

    def inject(log, queue):
        if not injected:
            if method == 'inject':
                queue.inject({**{key: values[0] for key, values in compiled.params().items()}, name: 'false'}, prioritize=True)
            else:
                assert queue.inject_value(name, 'false')
            injected.append(True)

    monkeypatch.setattr(LogRegBinary, 'train', fit)
    loop = UniversalExperimentLoop(sfd=compiled, data=bars, experiment_dir=tmp_path / 'injected',
        yaml_reference=config, search_strategy=build_search_strategy(config), feedback_interval=1,
        checkpoint_interval=1, intra_callback=inject)
    with pytest.raises(ValueError, match=rf'{name} must be a bool'):
        loop.run('injected_control', n_permutations=3, prep_each_round=True, progress_bar=False)
    assert calls == [True, True]
    assert injected == [True]


@pytest.mark.parametrize('name', ['strict_mode', 'use_calibration', 'use_threshold'])
def test_public_validate_rejects_quoted_false(tmp_path, name):
    config = _config('logreg', 'sigmoid')
    if name == 'strict_mode':
        config['sfd']['manifest'][name] = 'false'
    else:
        config['sfd']['params'][name] = ['false']
    stream = StringIO()
    round_trip_yaml().dump(config, stream)
    path = tmp_path / 'invalid.yaml'
    path.write_text(stream.getvalue())
    result = CliRunner().invoke(cli, ['validate', str(path)])
    assert result.exit_code == 1
    assert name in result.output and 'must be a bool' in result.output
