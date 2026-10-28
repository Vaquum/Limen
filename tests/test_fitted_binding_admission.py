from functools import partial
import json
from pathlib import Path
import shutil

from click.testing import CliRunner
import polars as pl
import pytest
from ruamel.yaml import YAML

from limen.cli.main import cli
from limen.experiment import UniversalExperimentLoop
from limen.experiment.manifest_core import DataSourceResolver
from limen.yaml import CompiledSFD, build_search_strategy, parse, validate
from limen.yaml import _fitted_binding as binding
from limen.yaml.store import commit_manifest
from tests.test_column_shims import FIXTURE
from tests.test_record_model_outputs import _config as _recorded_config


CASES = (('target', 'fit_params', 'quantil'), ('target', 'transform_params', 'shiftt'),
         ('scaler', 'params', 'qq'), ('calibration', 'probability_calibration', 'methdo'),
         ('calibration', 'threshold_function', 'threshold_stpe'))


def _config():
    config = _recorded_config(calibrated=True)
    config['sfd']['manifest']['data_source'] = {'method': 'limen.data.HistoricalData.get_any_file',
        'params': {'file_path_or_url': str(FIXTURE)}}
    return config


def _defect(config, case):
    section, part, key = case
    block = config['sfd']['manifest'][section]
    values = block.setdefault(part, {}) if section != 'calibration' else block[part].setdefault('params', {})
    values[key] = 3
    return f'sfd.manifest.{section}.{part}' + ('.params' if section == 'calibration' else '') + f'.{key}'


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('command', [['validate'], ['profile'], ['run', '--dry-run'], ['run']])
def test_fitted_typos_refuse_each_fresh_cli_surface_before_io(tmp_path, monkeypatch, case, command):
    config = _config()
    path = _defect(config, case)
    assert any(e.path == path for e in validate(config).errors)
    assert not any(e.path == 'sfd.manifest.scaler.params' for e in validate(config).errors)
    submitted = tmp_path / 'fitted.yaml'
    YAML().dump(config, submitted)
    (tmp_path / 'limen.toml').write_text('')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    def forbidden(*args, **kwargs):
        pytest.fail('Invalid fitted declaration reached source work')

    monkeypatch.setattr(DataSourceResolver, 'resolve', forbidden)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, [*command, str(submitted)])
    assert result.exit_code == 1 and path in result.output
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('dry_run', [False, True])
def test_legacy_committed_fitted_typo_refuses_without_store_writes(tmp_path, monkeypatch, case, dry_run):
    config = _config()
    path = _defect(config, case)
    submitted = tmp_path / 'fitted.yaml'
    YAML().dump(config, submitted)
    (tmp_path / 'limen.toml').write_text('')
    monkeypatch.chdir(tmp_path)
    identity, _ = commit_manifest(submitted, project_root=tmp_path)
    before = {str(p.relative_to(tmp_path)): (p.read_bytes(), p.stat().st_mtime_ns)
              for p in tmp_path.rglob('*') if p.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail('Invalid committed fitted declaration reached source work')

    monkeypatch.setattr(DataSourceResolver, 'resolve', forbidden)
    result = CliRunner().invoke(cli, ['run', *(['--dry-run'] if dry_run else []), 'manifest://' + identity])
    assert result.exit_code == 1 and path in result.output
    assert {str(p.relative_to(tmp_path)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in tmp_path.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize(('section', 'part', 'key'), [
    ('target', 'fit_params', 'train_data'), ('target', 'fit_params', 'target_name'),
    ('target', 'transform_params', 'data'),
    *[('scaler', 'params', key) for key in ('data', '_cls', '_p', 'x_train')],
    *[('calibration', part, key) for part in ('probability_calibration', 'threshold_function') for key in ('self', 'func')],
    ('calibration', 'probability_calibration', 'x_val'), ('calibration', 'probability_calibration', 'y_val'),
    ('calibration', 'threshold_function', 'y_val'), ('calibration', 'threshold_function', 'val_proba'),
    ('calibration', 'threshold_function', '_objective_maximize'),
])
def test_framework_and_callee_operand_collisions_are_located(section, part, key):
    config = _config()
    path = _defect(config, (section, part, key))
    assert any(e.path == path for e in validate(config).errors)


@pytest.mark.parametrize('section', ['fit_params', 'transform_params', 'scaler', 'probability_calibration', 'threshold_function'])
def test_nonstring_fitted_keywords_refuse_without_reference_scan_crash(section):
    config = _config()
    manifest = config['sfd']['manifest']
    if section in ('fit_params', 'transform_params'):
        block, prefix = manifest['target'].setdefault(section, {}), f'sfd.manifest.target.{section}'
    elif section == 'scaler':
        block, prefix = manifest['scaler'].setdefault('params', {}), 'sfd.manifest.scaler.params'
    else:
        block, prefix = manifest['calibration'][section].setdefault('params', {}), f'sfd.manifest.calibration.{section}.params'
    block[1] = 2
    assert any(e.path == prefix + '.1' for e in validate(config).errors)


@pytest.mark.parametrize('value', ['limen.scalers.Nope', 'limen.scalers.registry.SCALER_REGISTRY', 'limen.indicators.roc', None, 1])
def test_explicit_scaler_requires_a_resolved_class(value):
    config = _config()
    config['sfd']['manifest']['scaler'] = {'class': value}
    assert any(e.path == 'sfd.manifest.scaler.class' for e in validate(config).errors)


@pytest.mark.parametrize('choices', [None, ['nope'], [1]])
def test_from_params_requires_declared_valid_registry_candidates(choices):
    config = _config()
    config['sfd']['manifest']['scaler'] = {'from_params': 'scaler_type'}
    if choices is not None:
        config['sfd']['params']['scaler_type'] = choices
    result = validate(config)
    assert any(e.path == ('sfd.manifest.scaler.from_params' if choices is None else 'sfd.params.scaler_type[0]') for e in result.errors)


def test_all_scaler_candidates_bind_once_even_if_never_selected(monkeypatch):
    config = _config()
    config['sfd']['manifest']['scaler'] = {'from_params': 'scaler_type', 'params': {'quantile_range': [25, 75]}}
    config['sfd']['params']['scaler_type'] = ['robust', 'logreg', 'robust']
    original = binding.binding_errors
    seen = []

    def observe(component, args, kwargs):
        seen.append(component)
        return original(component, args, kwargs)

    monkeypatch.setattr(binding, 'binding_errors', observe)
    errors = binding.fitted_errors(config)
    assert any(e.path == 'sfd.manifest.scaler.params.quantile_range' and 'logreg' in e.message and 'LogRegScaler' in e.message for e in errors)
    assert seen.count(binding.SCALER_REGISTRY['robust']) == seen.count(binding.SCALER_REGISTRY['logreg']) == 1
    config['sfd']['manifest']['scaler']['params'] = {'scaler_type': 'linear'}
    assert any(e.path == 'sfd.manifest.scaler.params.scaler_type' for e in binding.fitted_errors(config))


@pytest.mark.parametrize(('flag', 'part', 'key'), [('use_calibration', 'probability_calibration', 'methdo'), ('use_threshold', 'threshold_function', 'threshold_stpe')])
def test_inactive_declared_calibration_still_requires_valid_binding(flag, part, key):
    config = _config()
    config['sfd']['params'][flag] = [False]
    path = _defect(config, ('calibration', part, key))
    assert any(e.path == path for e in validate(config).errors)


def test_declared_calibration_requires_architecture_keyword_acceptance():
    config = _config()
    config['sfd']['manifest']['reference_architecture'] = 'limen.sfd.reference_architecture.ridge_regressor'
    assert any(e.path == 'sfd.manifest.calibration' and 'prediction_calibration_config' in e.message for e in binding.fitted_errors(config))


@pytest.mark.parametrize(('architecture', 'accepted'), [
    (lambda *prediction_calibration_config: None, False),
    (lambda prediction_calibration_config, /: None, False),
    (lambda prediction_calibration_config: None, True),
    (lambda *, prediction_calibration_config: None, True),
    (lambda **prediction_calibration_config: None, True),
    (lambda *prediction_calibration_config, **kwargs: None, True),
])
def test_calibration_architecture_parameter_kind_matches_real_keyword_call(monkeypatch, architecture, accepted):
    config = _config()
    original = binding._resolve

    def resolve(name, path, errors, **kwargs):
        return architecture if path.endswith('reference_architecture') else original(name, path, errors, **kwargs)

    monkeypatch.setattr(binding, '_resolve', resolve)
    errors = binding.fitted_errors(config)
    assert not errors if accepted else any(e.path == 'sfd.manifest.calibration' for e in errors)
    if accepted:
        architecture(prediction_calibration_config=object())
    else:
        with pytest.raises(TypeError):
            architecture(prediction_calibration_config=object())


@pytest.mark.parametrize('key', ['metric', '_objective_maximize', 'y_val', 'val_proba'])
def test_objective_threshold_operands_are_framework_owned(key):
    config = _config()
    manifest = config['sfd']['manifest']
    manifest['objective'] = {'metric': 'backtest_total_return', 'direction': 'maximize'}
    del manifest['calibration']['threshold_function']['params']['metric']
    assert not binding.fitted_errors(config)
    path = _defect(config, ('calibration', 'threshold_function', key))
    assert any(e.path == path for e in binding.fitted_errors(config))


@pytest.mark.parametrize('part', ['fit_params', 'transform_params'])
def test_target_trade_context_is_supplied_and_reserved(part):
    config = _config()
    target = config['sfd']['manifest']['target']
    target.update({'class': 'limen.targets.TradeOutcomeTarget', 'fit_params': {}, 'transform_params': {}})
    assert not binding.fitted_errors(config)
    target[part]['trade_context'] = 'caller_context'
    assert any(e.path == f'sfd.manifest.target.{part}.trade_context' for e in binding.fitted_errors(config))
    target[part].clear()
    target['transform_params']['body_level_option'] = 1
    assert not binding.fitted_errors(config)


def test_target_missing_required_parameters_names_the_fit_envelope():
    config = _config()
    config['sfd']['manifest']['target']['fit_params'] = {}
    assert any(e.path == 'sfd.manifest.target.fit_params' and 'source_column' in e.message and 'quantile' in e.message
               for e in binding.fitted_errors(config))


def test_objective_calibration_wrapper_checks_apply_to_a_variadic_callee(monkeypatch):
    config = _config()
    manifest = config['sfd']['manifest']
    manifest['objective'] = {'metric': 'backtest_total_return', 'direction': 'maximize'}
    del manifest['calibration']['threshold_function']['params']['metric']
    original = binding._resolve

    def accepts(*args, **kwargs):
        pytest.fail('Static admission executed a calibrator')

    def resolve(name, path, errors, **kwargs):
        return accepts if path.endswith('probability_calibration.func') else original(name, path, errors, **kwargs)

    monkeypatch.setattr(binding, '_resolve', resolve)
    path = _defect(config, ('calibration', 'probability_calibration', 'clf'))
    assert any(e.path == path for e in binding.fitted_errors(config))


def test_opaque_architecture_acceptance_stays_explicitly_indeterminate(monkeypatch, caplog):
    config = _config()
    original = binding._resolve

    def architecture(*args, **kwargs):
        pytest.fail('Static admission executed an architecture')

    architecture.__signature__ = 'opaque'

    def resolve(name, path, errors, **kwargs):
        return architecture if path.endswith('reference_architecture') else original(name, path, errors, **kwargs)

    monkeypatch.setattr(binding, '_resolve', resolve)
    with caplog.at_level('DEBUG', logger=binding.__name__):
        assert not binding.fitted_errors(config)
    assert 'architecture acceptance is indeterminate' in caplog.text


def test_objective_without_declared_calibration_preserves_its_architecture_allowance():
    config = _config()
    manifest = config['sfd']['manifest']
    del manifest['calibration']
    manifest['objective'] = {'metric': 'backtest_total_return', 'direction': 'maximize'}
    manifest['reference_architecture'] = 'limen.sfd.reference_architecture.ridge_regressor'
    assert not binding.fitted_errors(config)


def test_target_transform_descriptor_forms_and_opaque_signature_never_construct():
    class Target:
        def __init__(self, train_data, target_name):
            pytest.fail('Static binding constructed a target')

        def transform(self, data, *, period=1):
            pytest.fail('Static binding executed a target')

    original = binding._resolve
    from unittest.mock import patch
    with patch.object(binding, '_resolve', return_value=Target):
        for transform in (Target.transform, staticmethod(lambda data, *, period=1: data), classmethod(lambda cls, data, *, period=1: data)):
            Target.transform = transform
            errors = []
            binding._target({'class': 'unused', 'transform_params': {'period': 1}}, errors)
            assert not errors
        Target.transform = property(lambda self: pytest.fail('Static binding evaluated a property'))
        errors = []
        binding._target({'class': 'unused'}, errors)
        assert not errors
    assert binding._resolve is original


def test_private_binding_preserves_required_kwargs_partials_and_indeterminate():
    def component(frame, *, required, optional=1, **kwargs):
        pytest.fail('Binding executed a component')

    errors = []
    binding._check(component, (object(),), {}, 'params', errors)
    assert len(errors) == 1 and errors[0].path == 'params' and 'required' in errors[0].message
    errors = []
    binding._check(partial(component, required=1), (object(),), {'required': 2, 'other': 3}, 'params', errors)
    assert not errors
    component.__signature__ = 'opaque'
    binding._check(component, (object(),), {'other': 3}, 'params', errors)
    assert not errors


def test_actual_shipped_fitted_binding_corpus(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    original = binding.binding_errors
    outcomes = []
    counts = dict.fromkeys(('files', 'target', 'scaler', 'probability_calibration', 'threshold_function'), 0)

    def observe(component, args, kwargs):
        result = original(component, args, kwargs)
        outcomes.append(result)
        return result

    monkeypatch.setattr(binding, 'binding_errors', observe)
    for directory in ('limen/yaml/templates', 'docs/examples', 'tests/fixtures'):
        for path in sorted((root / directory).rglob('*.yaml')):
            document, errors = parse(path)
            if errors or not isinstance(document.get('sfd'), dict):
                continue
            manifest = document['sfd'].get('manifest')
            if not isinstance(manifest, dict):
                continue
            counts['files'] += 1
            for kind in ('target', 'scaler'):
                counts[kind] += manifest.get(kind) is not None
            for kind in ('probability_calibration', 'threshold_function'):
                counts[kind] += (manifest.get('calibration') or {}).get(kind) is not None
            assert not binding.fitted_errors(document), path
    assert counts == {'files': 85, 'target': 72, 'scaler': 67, 'probability_calibration': 11, 'threshold_function': 13}
    assert len(outcomes) >= 220 and all(outcome == {} for outcome in outcomes)


@pytest.fixture(scope='module')
def genuine_saved_fitted_run(tmp_path_factory):
    root = tmp_path_factory.mktemp('fitted-recorded')
    config = _config()
    config['sfd']['params']['C'] = [0.1, 0.3, 0.5, 1.0]
    config['uel']['n_permutations'] = 4
    directory = root / 'saved'
    compiled = CompiledSFD(config)

    def stop(log, _queue):
        if log.height == 2:
            loop._shutdown_requested = True

    loop = UniversalExperimentLoop(sfd=compiled, data=compiled.manifest().fetch_data(), experiment_dir=directory,
        search_strategy=build_search_strategy(config), yaml_reference=config, feedback_interval=1,
        checkpoint_interval=1, intra_callback=stop)
    loop.run('fitted_admission', n_permutations=4, prep_each_round=True, progress_bar=False,
        recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True))
    log = pl.read_csv(directory / 'results.csv')
    assert loop.interrupted and log.height == 2 and log['strict_mode_error'].null_count() == 2
    assert (directory / 'checkpoint.json').is_file()
    return directory


@pytest.mark.parametrize('case', CASES)
def test_saved_fitted_typo_refuses_before_checkpoint_source_or_writes(tmp_path, monkeypatch, genuine_saved_fitted_run, case):
    directory = tmp_path / 'saved'
    shutil.copytree(genuine_saved_fitted_run, directory)
    metadata_path = directory / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    path = _defect(metadata['yaml_reference'], case)
    metadata_path.write_text(json.dumps(metadata))
    before = {str(p.relative_to(directory)): (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.rglob('*') if p.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail('Invalid saved fitted declaration reached reconstruction')

    monkeypatch.setattr(DataSourceResolver, 'resolve', forbidden)
    from limen.experiment.checkpoint_manager import CheckpointManager
    monkeypatch.setattr(CheckpointManager, 'load', forbidden)
    result = CliRunner().invoke(cli, ['run', '--resume', str(directory)])
    assert result.exit_code == 1 and path in result.output
    assert {str(p.relative_to(directory)): (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.rglob('*') if p.is_file()} == before


def test_valid_saved_fitted_run_preserves_completed_prefix(tmp_path, monkeypatch, genuine_saved_fitted_run):
    directory = tmp_path / 'saved'
    shutil.copytree(genuine_saved_fitted_run, directory)
    before = pl.read_csv(directory / 'results.csv').drop('execution_time')
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ['run', '--resume', str(directory)])
    assert result.exit_code == 0, result.output
    after = pl.read_csv(directory / 'results.csv').drop('execution_time')
    assert after.height == 4 and after.head(2).equals(before)


@pytest.mark.parametrize(('flag', 'part', 'key'), [('use_calibration', 'probability_calibration', 'methdo'), ('use_threshold', 'threshold_function', 'threshold_stpe')])
def test_genuinely_recorded_inactive_legacy_binding_refuses_resume_without_rewriting(tmp_path, monkeypatch, flag, part, key):
    config = _config()
    config['sfd']['params'].update(C=[0.1, 0.3], **{flag: [False, True]})
    config['uel']['n_permutations'] = 4
    path = _defect(config, ('calibration', part, key))
    compiled = CompiledSFD(config)
    directory = tmp_path / 'legacy'

    def stop(log, _queue):
        if log.height == 2:
            loop._shutdown_requested = True

    loop = UniversalExperimentLoop(sfd=compiled, data=compiled.manifest().fetch_data(), experiment_dir=directory,
        search_strategy=build_search_strategy(config), yaml_reference=config, feedback_interval=1,
        checkpoint_interval=1, intra_callback=stop)
    loop.run('inactive_fitted', n_permutations=4, prep_each_round=True, progress_bar=False,
        recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True))
    log = pl.read_csv(directory / 'results.csv')
    assert loop.interrupted and log.height == 2 and log[flag].to_list() == [False, False]
    assert log['strict_mode_error'].null_count() == 2 and (directory / 'checkpoint.json').is_file()
    saved = json.loads((directory / 'metadata.json').read_text())['yaml_reference']
    assert saved['sfd']['manifest']['calibration'][part]['params'][key] == 3
    before = {str(p.relative_to(directory)): (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.rglob('*') if p.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail('Invalid legacy fitted declaration reached reconstruction')

    monkeypatch.setattr(DataSourceResolver, 'resolve', forbidden)
    monkeypatch.setattr(CompiledSFD, '__init__', forbidden)
    monkeypatch.setattr('limen.experiment.checkpoint_manager.CheckpointManager.load', forbidden)
    result = CliRunner().invoke(cli, ['run', '--resume', str(directory)])
    assert result.exit_code == 1 and path in result.output
    assert {str(p.relative_to(directory)): (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.rglob('*') if p.is_file()} == before
