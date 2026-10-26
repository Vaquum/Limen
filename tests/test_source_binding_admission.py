from functools import partial
from copy import deepcopy
import json
import shutil

from click.testing import CliRunner
import polars as pl
import pytest
from ruamel.yaml import YAML

from limen.cli.main import cli
from limen.experiment import UniversalExperimentLoop
from limen.experiment.manifest_core import DataSourceResolver
from limen.experiment._source_contract import source_parameters
from limen.yaml import CompiledSFD, build_search_strategy, validate
from limen.yaml.store import commit_manifest
from limen.yaml._source_binding import source_call_errors
from tests.test_column_shims import _assert_prepared_equal, _config, FIXTURE
from tests.test_manifest_data_sources import _NeedsArgument, _Source, _recorded
from tests.test_fixed_trade_sources import _config as _trade_config


@pytest.mark.parametrize('command', [['validate'], ['profile'], ['run', '--dry-run'], ['run']])
@pytest.mark.parametrize('defect', ['class', 'namespace', 'keyword'])
def test_fresh_yaml_and_cli_refuse_before_source_or_artifacts(tmp_path, monkeypatch, defect, command):
    config = _config()
    source = config['sfd']['manifest']['data_source']
    if defect == 'class':
        source['method'] = 'limen.data.HistoricalData'
        path = 'sfd.manifest.data_source.method'
    elif defect == 'namespace':
        source['method'] = 'os.system'
        path = 'sfd.manifest.data_source.method'
    else:
        source['params']['row_cout_limit'] = 3
        path = 'sfd.manifest.data_source.params.row_cout_limit'
    assert any(error.path == path for error in validate(config).errors)
    submitted = tmp_path / 'source.yaml'
    YAML().dump(config, submitted)
    (tmp_path / 'limen.toml').write_text('')
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError('Invalid source declaration reached fetching')

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(DataSourceResolver, 'resolve', forbidden)
    result = CliRunner().invoke(cli, [*command, str(submitted)])
    assert result.exit_code == 1 and path in result.output
    assert not calls and not (tmp_path / 'results').exists()


@pytest.fixture(scope='module')
def genuine_saved_source_run(tmp_path_factory):
    root = tmp_path_factory.mktemp('source-admission-recorded')
    config = _config()
    config['sfd']['params']['alpha'] = [0.1, 0.2, 0.3, 0.4]
    config['uel'].update(n_permutations=4, checkpoint_interval=1)
    directory = root / 'saved'
    compiled = CompiledSFD(config)
    raw = compiled.manifest().fetch_data()

    def stop(log, _queue):
        if log.height == 2:
            loop._shutdown_requested = True

    loop = UniversalExperimentLoop(sfd=compiled, data=raw, experiment_dir=directory,
        search_strategy=build_search_strategy(config), yaml_reference=config,
        feedback_interval=1, checkpoint_interval=1, intra_callback=stop)
    loop.run('source_admission', n_permutations=4, prep_each_round=True, progress_bar=False,
        recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True))
    log = pl.read_csv(directory / 'results.csv')
    assert loop.interrupted and log.height == 2 and log['strict_mode_error'].null_count() == 2
    assert (directory / 'checkpoint.json').is_file()
    return directory


@pytest.mark.parametrize('role', ['learning', 'execution', 'funding'])
@pytest.mark.parametrize('defect', ['class', 'namespace', 'keyword'])
def test_genuine_saved_source_refuses_before_checkpoint_or_fetch_and_preserves_evidence(tmp_path, monkeypatch, genuine_saved_source_run, defect, role):
    directory = tmp_path / 'saved'
    shutil.copytree(genuine_saved_source_run, directory)
    metadata_path = directory / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    manifest = metadata['yaml_reference']['sfd']['manifest']
    source = {'method': 'limen.data.HistoricalData.get_any_file', 'params': {'file_path_or_url': str(FIXTURE)}}
    if role == 'learning':
        manifest['data_source'] = source
        prefix = 'sfd.manifest.data_source'
    elif role == 'execution':
        manifest['backtest']['execution_data_source'] = source
        prefix = 'sfd.manifest.backtest.execution_data_source'
    else:
        manifest['backtest']['funding'] = {'preset': 'binance_btcusdt', 'data_source': source}
        prefix = 'sfd.manifest.backtest.funding.data_source'
    if defect == 'class':
        source['method'] = 'limen.data.HistoricalData'
    elif defect == 'namespace':
        source['method'] = 'os.system'
    else:
        source['params']['row_cout_limit'] = 3
    metadata_path.write_text(json.dumps(metadata))
    before = {str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns) for path in directory.rglob('*') if path.is_file()}
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError('Saved source defect reached checkpoint or fetching')

    monkeypatch.setattr(DataSourceResolver, 'resolve', forbidden)
    monkeypatch.setattr('limen.experiment.checkpoint_manager.CheckpointManager.load', forbidden)
    result = CliRunner().invoke(cli, ['run', '--resume', str(directory), '--no-progress-bar'])
    assert result.exit_code == 1 and prefix in result.output
    assert not calls
    assert {str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns) for path in directory.rglob('*') if path.is_file()} == before


def test_source_call_binding_preserves_existing_receivers_partials_and_constructor_limits():
    frame = _recorded()
    owner = _Source()
    for method, params in ((_Source.load, {'rows': 3}), (owner.load, {'rows': 3}),
        (_Source.static, {'rows': 3}), (_Source.named_self, {'self': frame, 'rows': 3}),
        (owner, {'rows': 3}), (partial(_recorded, rows=2), {'rows': 3}),
        (partial(pl.read_parquet, FIXTURE), {'n_rows': 3})):
        assert source_call_errors(method, params, None) == []
    errors = source_call_errors(_NeedsArgument.load, {}, None)
    assert len(errors) == 1 and errors[0].path.endswith('.method') and 'required' in errors[0].message


def test_source_binding_uses_actual_compiler_date_injection_without_invocation():
    calls = []

    def source(*, start_date_limit, end_date_limit, **kwargs):
        calls.append(1)
        raise AssertionError('Static binding executed source')

    dates = {'train_start': '2024-01-01', 'test_end': '2024-01-12'}
    params = {'row_count_limit': 3}
    assert source_parameters(source, params, dates) == {**params,
        'start_date_limit': dates['train_start'], 'end_date_limit': dates['test_end']}
    assert source_call_errors(source, params, dates) == []
    missing = source_call_errors(source, params, None)
    assert len(missing) == 1 and 'start_date_limit' in missing[0].message and 'end_date_limit' in missing[0].message
    assert not calls and params == {'row_count_limit': 3}


def test_uninspectable_source_remains_indeterminate_without_invocation():
    class Source:
        @property
        def __signature__(self):
            raise ValueError('Signature deliberately unavailable')

        def __call__(self):
            raise AssertionError('Indeterminate source executed')

    assert source_call_errors(Source(), {}, None) == []


@pytest.mark.parametrize('missing', ['train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end'])
@pytest.mark.parametrize('command', [['validate'], ['profile'], ['run', '--dry-run'], ['run']])
def test_incomplete_dates_report_split_errors_without_source_or_artifacts(tmp_path, monkeypatch, missing, command):
    config = _config()
    manifest = config['sfd']['manifest']
    manifest['data_source'] = {'method': 'limen.data.HistoricalData.get_spot_klines', 'params': {}}
    del manifest['split_dates'][missing]
    expected = 'sfd.manifest.split_dates.' + missing
    assert any(error.path == expected for error in validate(config).errors)
    submitted = tmp_path / 'incomplete.yaml'
    YAML().dump(config, submitted)
    (tmp_path / 'limen.toml').write_text('')
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError('Incomplete dates reached source execution')

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(DataSourceResolver, 'resolve', forbidden)
    result = CliRunner().invoke(cli, [*command, str(submitted)])
    assert result.exit_code == 1 and expected in result.output
    assert not calls and not (tmp_path / 'results').exists()


@pytest.mark.parametrize('base_path', [None, 'not-selected.parquet'])
def test_source_overrides_supply_and_replace_required_arguments_with_recorded_equivalence(base_path):
    baseline = CompiledSFD(_config()).manifest()
    config = _config()
    manifest = config['sfd']['manifest']
    manifest['data_source']['params'] = {} if base_path is None else {'file_path_or_url': base_path}
    manifest['params_override'] = {'file_path_or_url': str(FIXTURE)}
    before = deepcopy(config)
    result = validate(config)
    assert result.valid, result.errors
    actual = CompiledSFD(config).manifest()
    assert actual.data_source_config.params == baseline.data_source_config.params
    raw = baseline.fetch_data()
    assert actual.fetch_data().equals(raw)
    _assert_prepared_equal(actual.prepare_data(raw, {}), baseline.prepare_data(raw, {}))
    assert config == before


@pytest.mark.parametrize('command', [['validate'], ['profile'], ['run', '--dry-run'], ['run']])
def test_required_source_override_passes_fresh_cli(tmp_path, monkeypatch, command):
    config = _config()
    manifest = config['sfd']['manifest']
    manifest['data_source']['params'] = {}
    manifest['params_override'] = {'file_path_or_url': str(FIXTURE)}
    submitted = tmp_path / 'override.yaml'
    YAML().dump(config, submitted)
    (tmp_path / 'limen.toml').write_text('')
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, [*command, str(submitted)])
    assert result.exit_code == 0, result.output


def test_override_binding_uses_compiler_precedence_and_excludes_split_config():
    calls = []

    def source(*, start_date_limit, end_date_limit):
        calls.append(1)
        raise AssertionError('Static admission executed a source')

    dates = {'train_start': '2024-01-01', 'test_end': '2024-01-12'}
    assert source_call_errors(source, {}, dates, overrides={'start_date_limit': '2024-01-02', 'split_config': (1, 0, 0)}) == []
    errors = source_call_errors(source, {}, dates, overrides={'start_date_limti': '2024-01-02'})
    assert len(errors) == 1 and errors[0].path == 'sfd.manifest.params_override.start_date_limti'
    assert not calls


@pytest.mark.parametrize('opaque', [False, True])
def test_override_keyword_names_refuse_at_their_declaration_even_when_opaque(opaque):
    def source(**kwargs):
        pytest.fail('Invalid source override was invoked')

    if opaque:
        source.__signature__ = 'unavailable'
    errors = source_call_errors(source, {}, None, overrides={0: 'invalid'})
    assert len(errors) == 1 and errors[0].path == 'sfd.manifest.params_override.0'


@pytest.mark.parametrize('opaque', [False, True])
@pytest.mark.parametrize('native_dispatch', [False, True])
def test_source_keyword_names_refuse_even_without_an_inspectable_signature(opaque, native_dispatch):
    def source(**kwargs):
        pytest.fail('Invalid keyword admission invoked a source')

    if opaque:
        source.__signature__ = 'unavailable'
    errors = source_call_errors(source, {1: 'invalid'}, None, native_dispatch=native_dispatch)
    assert len(errors) == 1 and errors[0].path == 'sfd.manifest.data_source.params.1'


def test_genuine_valid_saved_source_retains_its_completed_prefix(tmp_path, genuine_saved_source_run):
    directory = tmp_path / 'valid'
    shutil.copytree(genuine_saved_source_run, directory)
    before = pl.read_csv(directory / 'results.csv')
    result = CliRunner().invoke(cli, ['run', '--resume', str(directory), '--no-progress-bar'])
    assert result.exit_code == 0, result.output
    log = pl.read_csv(directory / 'results.csv')
    assert log.height == 4 and log['strict_mode_error'].null_count() == 4
    assert log.head(2).equals(before)


@pytest.mark.parametrize('role', ['execution', 'funding'])
@pytest.mark.parametrize('defect', ['class', 'namespace', 'keyword'])
@pytest.mark.parametrize('command', [['validate'], ['profile'], ['run', '--dry-run'], ['run']])
def test_secondary_loaders_refuse_all_fresh_cli_surfaces_before_source_or_artifacts(tmp_path, monkeypatch, role, defect, command):
    config = _trade_config(sources=False)
    manifest = config['sfd']['manifest']
    manifest['data_source'] = {'method': 'limen.data.HistoricalData.get_any_file', 'params': {'file_path_or_url': str(FIXTURE)}}
    source = {'method': 'limen.data.HistoricalData.get_any_file', 'params': {'file_path_or_url': str(FIXTURE)}}
    if defect == 'class':
        source['method'] = 'limen.data.HistoricalData'
    elif defect == 'namespace':
        source['method'] = 'os.system'
    else:
        source['params']['row_cout_limit'] = 3
    if role == 'execution':
        manifest['backtest']['execution_data_source'] = source
        prefix = 'sfd.manifest.backtest.execution_data_source'
    else:
        manifest['backtest']['funding'] = {'preset': 'binance_btcusdt', 'data_source': source}
        prefix = 'sfd.manifest.backtest.funding.data_source'
    expected = prefix + ('.params.row_cout_limit' if defect == 'keyword' else '.method')
    errors = validate(config).errors
    assert any(error.path == expected for error in errors)
    submitted = tmp_path / 'secondary.yaml'
    YAML().dump(config, submitted)
    (tmp_path / 'limen.toml').write_text('')
    before = {str(p.relative_to(tmp_path)): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError('Invalid secondary source reached data loading')

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(DataSourceResolver, 'resolve', forbidden)
    result = CliRunner().invoke(cli, [*command, str(submitted)])
    assert result.exit_code == 1 and expected in result.output
    assert not calls
    assert {str(p.relative_to(tmp_path)): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('role', ['learning', 'execution', 'funding'])
@pytest.mark.parametrize('defect', ['class', 'namespace', 'keyword'])
@pytest.mark.parametrize('command', [['run'], ['run', '--dry-run']])
def test_legacy_committed_loader_declarations_refuse_before_source_or_writes(tmp_path, monkeypatch, role, defect, command):
    config = _config() if role == 'learning' else _trade_config(sources=False)
    config['metadata']['mode'] = 'production'
    manifest = config['sfd']['manifest']
    manifest['data_source'] = {'method': 'limen.data.HistoricalData.get_any_file', 'params': {'file_path_or_url': str(FIXTURE)}}
    source = {'method': 'limen.data.HistoricalData.get_any_file', 'params': {'file_path_or_url': str(FIXTURE)}}
    if role == 'learning':
        manifest['data_source'] = source
        prefix = 'sfd.manifest.data_source'
    elif role == 'execution':
        manifest['backtest']['execution_data_source'] = source
        prefix = 'sfd.manifest.backtest.execution_data_source'
    else:
        manifest['backtest']['funding'] = {'preset': 'binance_btcusdt', 'data_source': source}
        prefix = 'sfd.manifest.backtest.funding.data_source'
    if defect == 'class':
        source['method'] = 'limen.data.HistoricalData'
    elif defect == 'namespace':
        source['method'] = 'os.system'
    else:
        source['params']['row_cout_limit'] = 3
    submitted = tmp_path / 'legacy.yaml'
    YAML().dump(config, submitted)
    (tmp_path / 'limen.toml').write_text('')
    identity, _ = commit_manifest(submitted, tmp_path)
    before = {str(p.relative_to(tmp_path)): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError('Stored source defect reached loading')

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(DataSourceResolver, 'resolve', forbidden)
    result = CliRunner().invoke(cli, [*command, 'manifest://' + identity])
    assert result.exit_code == 1 and prefix in result.output
    assert not calls
    assert {str(p.relative_to(tmp_path)): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()} == before
