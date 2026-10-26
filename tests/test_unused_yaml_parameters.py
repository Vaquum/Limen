import importlib
import inspect
import json
from pathlib import Path

import polars as pl
import pytest
from click.testing import CliRunner
from ruamel.yaml import YAML

from limen.cli.main import cli
from limen.experiment import ReuseContract, UniversalExperimentLoop, declare_reuse, _round_storage
from limen.inference import Trainer
from limen.yaml import CompiledSFD, build_search_strategy, parse, validate
from limen.yaml.rules import ParamCoverage
from limen.yaml.store import commit_manifest
from limen.yaml.validator import unused_errors

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet'


def _config():
    config, errors = parse(ROOT / 'docs/examples/compiled-research/recovery.yaml')
    assert not errors
    config['sfd']['manifest']['data_source']['params']['file_path_or_url'] = str(FIXTURE)
    assert validate(config).valid
    return config


def _snapshot(directory):
    return {str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob('*') if path.is_file()}


@pytest.mark.parametrize('values', [[1], [1, 2]])
def test_unused_axis_refuses_all_fresh_cli_boundaries_before_work(tmp_path, monkeypatch, values):
    config = _config()
    config['metadata']['mode'] = 'production'
    config['sfd']['params']['unused_control'] = values
    result = validate(config)
    assert not result.valid and not result.warnings
    assert [error.path for error in result.errors] == ['sfd.params.unused_control']
    assert f'{len(values)} declared choice(s)' in result.errors[0].message
    assert 'uel.search_strategy.seed' in result.errors[0].suggestion
    submitted = tmp_path / 'dead.yaml'
    YAML().dump(config, submitted)
    (tmp_path / 'limen.toml').write_text('')
    monkeypatch.chdir(tmp_path)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError('Unused axis reached compilation or source access')

    monkeypatch.setattr(CompiledSFD, '__init__', forbidden)
    monkeypatch.setattr('limen.experiment.manifest_core.DataSourceResolver.resolve', staticmethod(forbidden))
    before = _snapshot(tmp_path)
    for command in (['validate'], ['run', '--dry-run'], ['run', '--no-progress-bar'], ['profile'], ['commit']):
        outcome = CliRunner().invoke(cli, [*command, str(submitted)])
        assert outcome.exit_code == 1, outcome.output
        assert '[sfd.params.unused_control]' in outcome.output
        assert calls == [] and _snapshot(tmp_path) == before


@pytest.mark.parametrize('literal', ['group', 'source', 'override', 'escaped', 'native_nested'])
def test_literal_occurrences_do_not_credit_axes(literal):
    config = _config()
    manifest = config['sfd']['manifest']
    config['sfd']['params']['dead_axis'] = [1]
    if literal == 'group':
        manifest['indicators'][0]['params']['group'] = 'group_{dead_axis}'
    elif literal == 'source':
        manifest['data_source']['params']['file_path_or_url'] = 'recorded_{dead_axis}.parquet'
    elif literal == 'override':
        manifest['params_override'] = {'file_path_or_url': 'recorded_{dead_axis}.parquet'}
    elif literal == 'escaped':
        manifest['target']['fit_params']['source_column'] = 'roc_{{dead_axis}}'
    else:
        manifest['indicators'][0]['params']['literal_options'] = {'value': '{dead_axis}'}
    assert [error.path for error in unused_errors(config)] == ['sfd.params.dead_axis']


@pytest.mark.parametrize('site', ['execution', 'funding', 'barrier', 'allocation'])
def test_nested_resolve_json_references_consume_declared_axes(site):
    field = {'execution': 'execution_data_source', 'funding': 'funding', 'barrier': 'barrier_features'}
    params = {'nested': [{'value': '{axis}'}]}
    if site == 'allocation':
        manifest = {'allocation': {'method': 'limen.sfd.reference_architecture.meta_allocation.meta_allocation', 'params': params}}
    else:
        manifest = {'backtest': {field[site]: {'params': params}}}
    manifest['reference_architecture'] = 'limen.sfd.reference_architecture.logreg_binary'
    errors, warnings = [], []
    ParamCoverage().check({'sfd': {'manifest': manifest, 'params': {'axis': [1], 'dead_axis': [2]}}}, errors, warnings)
    assert not any(error.path == 'sfd.params.axis' for error in errors)
    if site != 'allocation':
        assert [error.path for error in errors] == ['sfd.params.dead_axis']
    assert not warnings


def test_unresolvable_consumer_cannot_prove_an_axis_unused():
    config = _config()
    config['sfd']['manifest']['reference_architecture'] = 'limen.sfd.reference_architecture.missing'
    config['sfd']['params']['unknown_consumer_axis'] = [1]
    result = validate(config)
    assert not result.valid and not any('Unused parameter' in error.message for error in result.errors)


def test_whitespace_in_json_source_values_remains_literal():
    from limen.experiment._resolve_trade_policy import resolve_json

    value = ' {axis} '
    assert resolve_json(value, {'axis': 'recorded.parquet'}) == value
    config = _config()
    config['sfd']['params']['axis'] = ['recorded.parquet']
    config['sfd']['manifest']['backtest']['execution_data_source'] = {
        'method': 'limen.data.HistoricalData.get_any_file', 'params': {'file_path_or_url': value}}
    assert [error.path for error in unused_errors(config)] == ['sfd.params.axis']


@pytest.mark.parametrize('template', ['roc_{}', 'roc_{0}', '{C!z}', 'roc_{{C}'])
def test_unsupported_native_formatting_is_indeterminate(template):
    from limen.experiment.manifest_core import _resolve_params

    config = _config()
    config['sfd']['manifest']['target']['fit_params']['source_column'] = template
    config['sfd']['params']['uncertain_axis'] = [1]
    params = {key: values[0] for key, values in config['sfd']['params'].items()}
    with pytest.raises((ValueError, IndexError)):
        _resolve_params({'source_column': template}, params)
    assert not unused_errors(config)


@pytest.mark.parametrize('contracted', [False, True])
def test_extension_consumption_preserves_recorded_preparation_and_model(monkeypatch, contracted):
    module = importlib.import_module('limen.sfd.reference_architecture.logreg_binary')
    observed = []

    def extension(data_dict, split_data, round_params, fitted_params):
        observed.append(round_params['extension_window'])
        for split in ('train', 'val', 'test'):
            data_dict['x_' + split] = data_dict['x_' + split].with_columns(
                pl.all().rolling_mean(window_size=round_params['extension_window'], min_samples=1))
        return data_dict

    extension.__module__ = module.__name__
    if contracted:
        declare_reuse(extension, ReuseContract(lambda settings: None,
            parameter_keys=('extension_window',), validation_keys=(), consumes=('extension_window',)))
    monkeypatch.setattr(module, 'parameter_extension', extension, raising=False)
    config = _config()
    config['sfd']['manifest']['data_dict_extension'] = {'func': module.__name__ + '.parameter_extension'}
    config['sfd']['params']['extension_window'] = [3]
    assert validate(config).valid
    config['sfd']['params']['dead_axis'] = [1]
    assert validate(config).valid is (not contracted)
    config['sfd']['params'].pop('dead_axis')
    compiled = CompiledSFD(config)
    params = {key: values[0] for key, values in compiled.params().items()}
    manifest = compiled.manifest()
    raw = pl.read_parquet(FIXTURE)
    manifest.data_dict_extension = None
    plain = manifest.prepare_data(raw, params)
    manifest.data_dict_extension = extension
    data = manifest.prepare_data(raw, params)
    for split in ('train', 'val', 'test'):
        assert data['x_' + split].equals(plain['x_' + split].with_columns(
            pl.all().rolling_mean(window_size=3, min_samples=1)))
    assert not data['x_test'].equals(plain['x_test'])
    result = manifest.run_model(data, params)
    assert observed == [3] and len(result['_preds']) == 95
    assert result['_model'].model.C == 0.1


def test_finite_kwargs_contract_credits_actual_model_inputs(monkeypatch):
    module = importlib.import_module('limen.sfd.reference_architecture.logreg_binary')
    consumed = tuple(name for name in inspect.signature(module.logreg_binary).parameters if name != 'data')

    def architecture(data, **params):
        return module.logreg_binary(data, **{key: value for key, value in params.items() if key in consumed})

    architecture.__module__ = module.__name__
    declare_reuse(architecture, ReuseContract(lambda settings: None, validation_keys=(), consumes=consumed))
    monkeypatch.setattr(module, 'finite_parameter_architecture', architecture, raising=False)
    config = _config()
    config['sfd']['manifest']['reference_architecture'] = module.__name__ + '.finite_parameter_architecture'
    assert validate(config).valid
    config['sfd']['params']['dead_axis'] = [1]
    assert [error.path for error in validate(config).errors] == ['sfd.params.dead_axis']
    config['sfd']['params'].pop('dead_axis')
    compiled = CompiledSFD(config)
    params = {key: values[0] for key, values in compiled.params().items()}
    manifest = compiled.manifest()
    data = manifest.prepare_data(pl.read_parquet(FIXTURE), params)
    result = manifest.run_model(data, params)
    assert result['_model'].model.C == 0.1 and len(result['_preds']) == 95


def test_var_positional_placeholder_is_not_a_keyword_consumer(monkeypatch):
    module = importlib.import_module('limen.sfd.reference_architecture.logreg_binary')

    def architecture(data, *args, C=1):
        return module.logreg_binary(data, C=C)

    architecture.__module__ = module.__name__
    monkeypatch.setattr(module, 'positional_parameter_architecture', architecture, raising=False)
    errors, warnings = [], []
    ParamCoverage().check({'sfd': {'manifest': {'reference_architecture': module.__name__ + '.positional_parameter_architecture'},
        'params': {'args': [1], 'C': [0.1]}}}, errors, warnings)
    assert [error.path for error in errors] == ['sfd.params.args'] and not warnings


@pytest.mark.parametrize('contracted', [False, True])
def test_allocation_context_consumes_recorded_axis(monkeypatch, contracted):
    from tests.test_meta_allocation import _config as allocation_config

    module = importlib.import_module('limen.sfd.reference_architecture.meta_allocation')
    observed = []

    def allocation(data, *, context, **params):
        observed.append(context.round_params['allocation_threshold'])
        return module.meta_allocation(data, context=context,
            **(params | {'probability_threshold': context.round_params['allocation_threshold']}))

    allocation.__module__ = module.__name__
    allocation.__qualname__ = 'context_parameter_allocation'
    if contracted:
        declare_reuse(allocation, ReuseContract(lambda settings: None, validation_keys=(),
            parameter_keys=('allocation_threshold',), consumes=('allocation_threshold',)))
    monkeypatch.setattr(module, 'context_parameter_allocation', allocation, raising=False)
    config = allocation_config()
    config['sfd']['manifest']['allocation']['method'] = module.__name__ + '.context_parameter_allocation'
    config['sfd']['params']['allocation_threshold'] = [0.1]
    assert validate(config).valid
    config['sfd']['params']['dead_axis'] = [1]
    assert validate(config).valid is (not contracted)
    config['sfd']['params'].pop('dead_axis')
    manifest = CompiledSFD(config).manifest()
    params = {key: values[0] for key, values in config['sfd']['params'].items()}
    data = manifest.prepare_data(pl.read_parquet(FIXTURE, n_rows=480), params)
    result = manifest.run_model(data, params)
    assert observed == [0.1] and result['_model'].threshold == 0.1
    assert result['allocation_training_completed'] >= 2 and len(result['_preds']) > 0


@pytest.mark.parametrize('dead', [False, True])
def test_genuine_saved_checkpoint_resume_and_assessment_preserve_evidence(tmp_path, monkeypatch, dead):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'limen.toml').write_text('')
    config = _config()
    if dead:
        config['sfd']['params']['unused_control'] = [1]
    directory = tmp_path / 'saved'
    raw = pl.read_parquet(FIXTURE)

    def stop(log, _msq):
        if log.height == 2:
            loop._shutdown_requested = True

    loop = UniversalExperimentLoop(sfd=CompiledSFD(config), data=raw, experiment_dir=directory,
        search_strategy=build_search_strategy(config), yaml_reference=config,
        feedback_interval=1, checkpoint_interval=1, intra_callback=stop)
    loop.run('legacy_unused', n_permutations=4, prep_each_round=True, progress_bar=False,
             recording=config['uel']['recording'])
    assert loop.interrupted and pl.read_csv(directory / 'results.csv').height == 2
    saved = json.loads((directory / 'metadata.json').read_text())['yaml_reference']
    assert ('unused_control' in saved['sfd']['params']) is dead
    with _round_storage.open_text(_round_storage.round_path(directory)) as stream:
        rounds = [json.loads(line) for line in stream if line.strip()]
    submitted = tmp_path / 'legacy.yaml'
    YAML().dump(config, submitted)
    before = _snapshot(tmp_path)
    assert Trainer(directory, data=raw).train([rounds[0]['round_id']])[0] is not None
    assert _snapshot(tmp_path) == before
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError('Unused saved axis reached checkpoint, construction or source')

    with monkeypatch.context() as guard:
        if dead:
            guard.setattr(CompiledSFD, '__init__', forbidden)
            guard.setattr('limen.experiment.checkpoint_manager.CheckpointManager.load', forbidden)
            guard.setattr('limen.experiment.manifest_core.DataSourceResolver.resolve', staticmethod(forbidden))
            assessment = CliRunner().invoke(cli, ['assess', str(submitted), '--results', str(directory)])
            assert assessment.exit_code == 1 and '[sfd.params.unused_control]' in assessment.output
            assert calls == [] and _snapshot(tmp_path) == before
        outcome = CliRunner().invoke(cli, ['run', '--resume', str(directory), '--no-progress-bar'])
    if dead:
        assert outcome.exit_code == 1 and 'sfd.params.unused_control' in outcome.output
        assert calls == [] and _snapshot(tmp_path) == before
    else:
        assert outcome.exit_code == 0, outcome.output
        assert pl.read_csv(directory / 'results.csv').height == 4


def test_legacy_stored_uri_refuses_without_rewriting_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'limen.toml').write_text('')
    config = _config()
    config['metadata']['mode'] = 'production'
    config['sfd']['params']['unused_control'] = [1, 2]
    submitted = tmp_path / 'legacy.yaml'
    YAML().dump(config, submitted)
    identity, _ = commit_manifest(submitted, tmp_path)
    before = _snapshot(tmp_path)
    for command in (['run'], ['run', '--dry-run']):
        outcome = CliRunner().invoke(cli, [*command, 'manifest://' + identity])
        assert outcome.exit_code == 1 and '[sfd.params.unused_control]' in outcome.output
        assert _snapshot(tmp_path) == before
