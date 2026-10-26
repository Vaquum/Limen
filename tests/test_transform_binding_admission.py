from functools import partial, wraps
import json
from pathlib import Path

from click.testing import CliRunner
import pytest
from ruamel.yaml import YAML

from limen import indicators
from limen.cli.main import cli
from limen.experiment import UniversalExperimentLoop
from limen.experiment.manifest_core import DataSourceResolver
from limen.features import lag_range, rolling_zscore
from limen.yaml import CompiledSFD, build_search_strategy, parse, validate
from limen.yaml.store import commit_manifest
from limen.yaml._callable_binding import binding_errors, check_transform_binding
from tests.test_column_shims import _config, FIXTURE


@pytest.mark.parametrize('kind', ['indicators', 'features', 'bar_formation', 'pre_split_data_selector'])
def test_every_transform_entry_rejects_misspelled_keyword_without_running_it(kind, monkeypatch):
    config = _config()
    entry = {'func': 'limen.indicators.roc', 'params': {'peroid': 3}}
    config['sfd']['manifest'][kind] = [entry] if kind in ('indicators', 'features') else entry
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError('Admission reached source work')

    monkeypatch.setattr(DataSourceResolver, 'resolve', forbidden)
    result = validate(config)
    path = f"sfd.manifest.{kind}" + ('[0]' if isinstance(config['sfd']['manifest'][kind], list) else '')
    assert not result.valid and any(e.path == path + '.params.peroid' for e in result.errors)
    assert not calls


@pytest.mark.parametrize('kind,key', [('indicators', 'func'), ('indicators', 'include_if'),
    ('indicators', 'function'), ('features', 'func'), ('features', 'include_if'),
    ('features', 'function'), ('bar_formation', 'func'), ('bar_formation', 'function'),
    ('pre_split_data_selector', 'func'), ('indicators', 'self'), ('features', 'self'),
    ('bar_formation', 'self'), ('pre_split_data_selector', 'self')])
def test_wrapper_collisions_name_the_exact_parameter(kind, key):
    config = _config()
    entry = {'func': 'limen.indicators.roc', 'params': {key: 'close'}}
    config['sfd']['manifest'][kind] = [entry] if kind in ('indicators', 'features') else entry
    prefix = f"sfd.manifest.{kind}" + ('[0]' if isinstance(config['sfd']['manifest'][kind], list) else '')
    assert any(e.path == prefix + '.params.' + key and 'wrapper' in e.message for e in validate(config).errors)


@pytest.mark.parametrize('function,params,missing', [
    ('limen.features.lag_range', {'col': 'close', 'start': 1}, 'end'),
    ('limen.features.rolling_zscore', {'column': 'close'}, 'window'),
])
def test_statically_missing_required_arguments_are_located(function, params, missing):
    config = _config()
    config['sfd']['manifest']['features'] = [{'func': function, 'params': params}]
    assert any(e.path == 'sfd.manifest.features[0].params' and missing in e.message for e in validate(config).errors)
    assert binding_errors(lag_range if function.endswith('lag_range') else rolling_zscore, (object(),), params)


def test_group_defaults_conditional_and_positional_frame_binding():
    config = _config()
    config['sfd']['params']['enabled'] = [False]
    entry = {'func': 'limen.indicators.roc', 'params': {'group': 'momentum'}, 'include_if': 'enabled'}
    config['sfd']['manifest']['indicators'] = [entry]
    assert validate(config).valid
    entry['params']['data'] = 'close'
    assert any(e.path == 'sfd.manifest.indicators[0].params.data' for e in validate(config).errors)
    del entry['params']['data']
    entry['params']['peroid'] = 3
    assert any(e.path.endswith('.params.peroid') for e in validate(config).errors)


def test_binding_helper_preserves_kwargs_partials_defaults_and_unknown_signatures():
    calls = []

    def accepts(frame, required, *, default=1, **kwargs):
        calls.append(frame)
        raise AssertionError('Signature admission executed callable')

    assert binding_errors(accepts, (object(),), {'required': 2, 'anything': 3}) == {}
    assert binding_errors(partial(accepts, required=2), (object(),), {}) == {}
    assert binding_errors(partial(accepts, required=2), (object(),), {'required': 3}) == {}
    assert binding_errors(accepts, (object(),), {}) == {'': 'Missing required arguments: required'}
    assert binding_errors(type, (object(),), {}) is None
    assert binding_errors(lambda: None, (object(),), {}) == {'': 'too many positional arguments'}
    assert not calls


def test_real_cli_refuses_typo_before_fetch_and_corrected_declaration_runs(tmp_path, monkeypatch):
    config = _config()
    config['uel']['output_path'] = 'binding_cli'
    config['sfd']['manifest']['indicators'] = [{'func': 'limen.indicators.roc', 'params': {'period': 3, 'peroid': 3}}]
    path = tmp_path / 'binding.yaml'
    YAML().dump(config, path)
    (tmp_path / 'limen.toml').write_text('')
    monkeypatch.chdir(tmp_path)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError('Invalid declaration fetched data')

    with monkeypatch.context() as guard:
        guard.setattr(DataSourceResolver, 'resolve', forbidden)
        guard.setattr(indicators, 'roc', wraps(indicators.roc)(forbidden))
        result = CliRunner().invoke(cli, ['run', '--no-progress-bar', str(path)])
    assert result.exit_code == 1 and 'indicators[0].params.peroid' in result.output
    assert not calls
    del config['sfd']['manifest']['indicators'][0]['params']['peroid']
    YAML().dump(config, path)
    result = CliRunner().invoke(cli, ['run', '--no-progress-bar', str(path)])
    assert result.exit_code == 0, result.output
    import polars as pl
    log = pl.read_csv(tmp_path / 'results/dev/binding_cli/results.csv')
    assert log.height == 1 and log['strict_mode_error'].null_count() == 1


def test_rule_based_indicator_binding_is_checked():
    config, errors = parse(FIXTURE.parents[2] / 'limen/yaml/templates/rule_based.yaml')
    assert not errors
    config['sfd']['manifest']['indicators'][0].setdefault('params', {})['peroid'] = 3
    assert any(e.path == 'sfd.manifest.indicators[0].params.peroid' for e in validate(config).errors)


def test_actual_shipped_example_fixture_transform_bindings_remain_admissible():
    root = Path(__file__).resolve().parents[1]
    checked = 0
    for directory in (root / 'limen/yaml/templates', root / 'docs/examples', root / 'tests/fixtures'):
        for path in sorted(directory.rglob('*.yaml')):
            config, parse_errors = parse(path)
            if parse_errors or not isinstance(config.get('sfd'), dict):
                continue
            manifest = config['sfd'].get('manifest')
            if not isinstance(manifest, dict):
                continue
            errors = []
            for kind in ('indicators', 'features'):
                for index, item in enumerate(manifest.get(kind) or []):
                    check_transform_binding(item, f'sfd.manifest.{kind}[{index}]', errors)
                    checked += 1
            for kind in ('bar_formation', 'pre_split_data_selector'):
                if (item := manifest.get(kind)) is not None:
                    check_transform_binding(item, f'sfd.manifest.{kind}', errors)
                    checked += 1
            assert not errors, (path, errors)
    assert checked >= 403


def test_selector_has_direct_call_binding_without_the_pipe_function_reservation():
    errors = []
    check_transform_binding({'func': 'limen.indicators.roc', 'params': {'function': 1}},
        'sfd.manifest.pre_split_data_selector', errors)
    assert len(errors) == 1 and errors[0].path.endswith('.params.function')
    assert 'unexpected keyword' in errors[0].message and 'wrapper' not in errors[0].message


def test_missing_arguments_are_checked_when_params_is_absent():
    config = _config()
    config['sfd']['manifest']['features'] = [{'func': 'limen.features.rolling_zscore'}]
    errors = validate(config).errors
    assert any(error.path == 'sfd.manifest.features[0].params' and 'column' in error.message and 'window' in error.message for error in errors)


def test_independent_binding_and_forward_reference_defects_both_report_when_inactive():
    config = _config()
    config['sfd']['params']['enabled'] = [False]
    config['sfd']['manifest']['indicators'] = [{'func': 'limen.indicators.roc',
        'params': {'period': '{missing_period}', 'peroid': 3}, 'include_if': 'enabled'}]
    errors = validate(config).errors
    assert any(error.path.endswith('.params.peroid') for error in errors)
    assert any(error.path.endswith('.params.period') and 'missing_period' in error.message for error in errors)


def test_nonstring_transform_keyword_is_located_without_a_reference_admission_crash():
    config = _config()
    config['sfd']['manifest']['indicators'] = [{'func': 'limen.indicators.roc', 'params': {1: 2}}]
    errors = validate(config).errors
    assert any(error.path == 'sfd.manifest.indicators[0].params.1' and 'must be strings' in error.message for error in errors)


def test_positional_only_frame_names_and_kwargs_keep_their_actual_binding_semantics():
    def positional(data, /, **options):
        raise AssertionError('Static binding executed transform')

    def keyword_frame(data, **options):
        raise AssertionError('Static binding executed transform')

    def no_frame(**options):
        raise AssertionError('Static binding executed transform')

    assert binding_errors(positional, (object(),), {'data': 1}) == {}
    assert 'data' in binding_errors(keyword_frame, (object(),), {'data': 1})
    assert '' in binding_errors(no_frame, (object(),), {})


def _snapshot(root):
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob('*') if path.is_file()}


@pytest.mark.parametrize('command', [['validate'], ['profile'], ['run', '--dry-run'], ['run']])
def test_all_fresh_cli_surfaces_refuse_before_source_or_writes(tmp_path, monkeypatch, command):
    config = _config()
    config['sfd']['manifest']['indicators'] = [{'func': 'limen.indicators.roc', 'params': {'peroid': 3}}]
    submitted = tmp_path / 'bad.yaml'
    YAML().dump(config, submitted)
    (tmp_path / 'limen.toml').write_text('')
    before = _snapshot(tmp_path)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError('Invalid transform reached source')

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(DataSourceResolver, 'resolve', forbidden)
    result = CliRunner().invoke(cli, [*command, str(submitted)])
    assert result.exit_code == 1 and 'sfd.manifest.indicators[0].params.peroid' in result.output
    assert not calls and _snapshot(tmp_path) == before


@pytest.mark.parametrize('command', [['run'], ['run', '--dry-run']])
def test_legacy_committed_uri_refuses_without_rewriting_identity(tmp_path, monkeypatch, command):
    config = _config()
    config['metadata']['mode'] = 'production'
    config['sfd']['manifest']['indicators'] = [{'func': 'limen.indicators.roc', 'params': {'peroid': 3}}]
    submitted = tmp_path / 'legacy.yaml'
    YAML().dump(config, submitted)
    (tmp_path / 'limen.toml').write_text('')
    identity, _ = commit_manifest(submitted, tmp_path)
    before = _snapshot(tmp_path)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError('Stored transform defect reached source')

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(DataSourceResolver, 'resolve', forbidden)
    result = CliRunner().invoke(cli, [*command, 'manifest://' + identity])
    assert result.exit_code == 1 and 'sfd.manifest.indicators[0].params.peroid' in result.output
    assert not calls and _snapshot(tmp_path) == before


def test_genuine_inactive_legacy_checkpoint_refuses_resume_before_reconstruction_or_writes(tmp_path, monkeypatch):
    config = _config()
    config['sfd']['params'].update(alpha=[0.1, 0.3], include_bad=[False, True])
    config['uel'].update(n_permutations=4, checkpoint_interval=1)
    config['sfd']['manifest']['indicators'] = [{'func': 'limen.indicators.roc',
        'params': {'period': 3, 'peroid': 3}, 'include_if': 'include_bad'}]
    compiled = CompiledSFD(config)
    raw = compiled.manifest().fetch_data()
    directory = tmp_path / 'legacy'

    def stop(log, _queue):
        if log.height == 2:
            loop._shutdown_requested = True

    loop = UniversalExperimentLoop(sfd=compiled, data=raw, experiment_dir=directory,
        search_strategy=build_search_strategy(config), yaml_reference=config,
        feedback_interval=1, checkpoint_interval=1, intra_callback=stop)
    loop.run('legacy_transform', n_permutations=4, prep_each_round=True, progress_bar=False,
        recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True))
    log = loop.experiment_log
    assert loop.interrupted and log.height == 2 and log['strict_mode_error'].null_count() == 2
    assert log['include_bad'].to_list() == [False, False]
    assert (directory / 'checkpoint.json').is_file()
    saved = json.loads((directory / 'metadata.json').read_text())['yaml_reference']
    assert saved['sfd']['manifest']['indicators'][0]['params']['peroid'] == 3
    before = _snapshot(directory)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError('Legacy transform defect reached checkpoint, compilation or source')

    monkeypatch.setattr(DataSourceResolver, 'resolve', forbidden)
    monkeypatch.setattr(CompiledSFD, '__init__', forbidden)
    monkeypatch.setattr('limen.experiment.checkpoint_manager.CheckpointManager.load', forbidden)
    result = CliRunner().invoke(cli, ['run', '--resume', str(directory), '--no-progress-bar'])
    assert result.exit_code == 1 and 'sfd.manifest.indicators[0].params.peroid' in result.output
    assert not calls and _snapshot(directory) == before


def test_outer_wrapper_and_keyword_name_errors_survive_an_uninspectable_callee(monkeypatch):
    monkeypatch.setattr(indicators.roc, '__signature__', 'unavailable', raising=False)
    errors = []
    check_transform_binding({'func': 'limen.indicators.roc',
        'params': {'self': 1, 'function': 1, 2: 3}}, 'sfd.manifest.indicators[0]', errors)
    assert {error.path for error in errors} == {'sfd.manifest.indicators[0].params.' + key for key in ('self', 'function', '2')}
