import json
from pathlib import Path
import subprocess
import sys

import polars as pl
import pytest
from click.testing import CliRunner

from limen.cli.main import cli
from limen.experiment import MLManifest
from limen.experiment.manifest_core import _process_bars, DataSourceResolver
from limen.yaml import CompiledSFD, build_manifest, parse, validate
from limen.yaml.config import round_trip_yaml

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/spot_15m_20250101_20250531.parquet'


def _config(kind='rule_based'):
    source = 'limen/yaml/templates/common_strategies.yaml' if kind == 'rule_based' else 'docs/examples/compiled-research/target_nextbarup.yaml'
    config, errors = parse(ROOT / source)
    assert not errors
    return config


@pytest.mark.parametrize('columns,error_type,index', [
    (None, TypeError, None), ([], ValueError, None), ('close', TypeError, None),
    ([''], ValueError, 0), ([' \t'], ValueError, 0), (['close', 'close'], ValueError, 1),
    ([True], TypeError, 0), ([['close']], TypeError, 0), ([1.0], TypeError, 0),
    ({'close': 1}, TypeError, None), (('close', 'volume'), TypeError, None),
])
def test_explicit_required_columns_refuse_at_both_declaration_boundaries(columns, error_type, index):
    config = _config()
    config['sfd']['manifest']['required_columns'] = columns
    result = validate(config)
    assert not result.valid and len(result.errors) == 1
    assert result.errors[0].path == 'sfd.manifest.required_columns' + (f'[{index}]' if index is not None else '')
    with pytest.raises(error_type, match='required_columns'):
        build_manifest(config)
    manifest = MLManifest().set_required_bar_columns(['close'])
    with pytest.raises(error_type, match='required_columns'):
        manifest.set_required_bar_columns(columns)
    assert manifest.required_bar_columns == ['close']


def test_generator_is_refused_without_consuming_it_or_changing_state():
    observed = []

    def columns():
        observed.append('iterated')
        yield 'volume'

    manifest = MLManifest().set_required_bar_columns(['close'])
    with pytest.raises(TypeError, match='must be a list'):
        manifest.set_required_bar_columns(columns())
    assert not observed and manifest.required_bar_columns == ['close']


def test_duplicate_reports_the_second_occurrence_and_original_index():
    config = _config()
    config['sfd']['manifest']['required_columns'] = ['close', 'volume', 'close']
    result = validate(config)
    assert [(error.path, error.message) for error in result.errors] == [
        ('sfd.manifest.required_columns[2]', "required_columns[2] duplicates required_columns[0] ('close')")]


@pytest.mark.parametrize('kind', ['ml', 'rule_based'])
def test_exact_names_and_owned_declarations_are_preserved(kind):
    config = _config(kind)
    columns = ['Close', 'close', ' close', 'volume']
    config['sfd']['manifest']['required_columns'] = columns
    assert validate(config).valid
    compiled = build_manifest(config)
    native = MLManifest().set_required_bar_columns(columns)
    assert compiled.required_bar_columns == native.required_bar_columns == columns
    assert compiled.required_bar_columns is not columns and native.required_bar_columns is not columns
    columns.append('close')
    assert compiled.required_bar_columns == native.required_bar_columns == ['Close', 'close', ' close', 'volume']
    bars = pl.read_parquet(FIXTURE, n_rows=200)
    with pytest.raises(AssertionError, match="Required bar column 'Close' not found after bar formation"):
        _process_bars(native, bars, {'bar_type': 'base'})


@pytest.mark.parametrize('formed', [False, True])
def test_omission_and_valid_declaration_preserve_recorded_base_and_formed_bars(formed):
    bars = pl.read_parquet(FIXTURE, n_rows=200)
    config = _config('ml')
    config['sfd']['manifest'].pop('required_columns', None)
    assert validate(config).valid
    compiled = build_manifest(config)
    assert compiled.required_bar_columns == []
    for manifest in (compiled, MLManifest().set_required_bar_columns(['datetime', 'close', 'volume'])):
        if formed:
            manifest.set_bar_formation(lambda frame: frame)
        datetimes, result = _process_bars(manifest, bars, {'bar_type': 'formed' if formed else 'base'})
        assert datetimes == bars['datetime'].to_list() and result.equals(bars)


@pytest.mark.parametrize('kind', ['ml', 'rule_based'])
@pytest.mark.parametrize('columns', [None, [], ['close', 'close'], [' \t']])
@pytest.mark.parametrize('command', [('validate',), ('run', '--dry-run'), ('run', '--no-progress-bar')])
def test_cli_refuses_invalid_declarations_before_fetch_or_artifact_writes(tmp_path, monkeypatch, kind, columns, command):
    monkeypatch.chdir(tmp_path)
    config = _config(kind)
    config['sfd']['manifest']['required_columns'] = columns
    path = tmp_path / 'columns.yaml'
    with path.open('w') as stream:
        round_trip_yaml().dump(config, stream)
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid columns reached reconstruction or source access')

    monkeypatch.setattr(CompiledSFD, '__init__', forbidden)
    monkeypatch.setattr(DataSourceResolver, 'resolve', staticmethod(forbidden))
    result = CliRunner().invoke(cli, [*command, str(path)])
    assert result.exit_code == 1 and 'ERROR  [sfd.manifest.required_columns' in result.output
    assert before == {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    assert not (tmp_path / 'results').exists()


@pytest.fixture(scope='module')
def optimized_admission():
    code = '''
import json
from limen.experiment import MLManifest
observed = []
def names():
    observed.append('iterated')
    yield 'volume'
results = []
for value in (None, [], 'close', ('close',), {'close': 1}, names(), [True], [['close']], [' '], ['close','close']):
    manifest = MLManifest().set_required_bar_columns(['volume'])
    try:
        manifest.set_required_bar_columns(value)
    except (TypeError, ValueError) as error:
        if manifest.required_bar_columns != ['volume']:
            raise RuntimeError('Invalid declaration changed stored columns')
        results.append(type(error).__name__)
    else:
        raise RuntimeError('Invalid declaration was accepted')
if observed:
    raise RuntimeError('Declaration consumed a generator')
print(json.dumps(results))
'''
    return {optimized: json.loads(subprocess.run([sys.executable, *(['-O'] if optimized else []), '-c', code],
            check=True, stdout=subprocess.PIPE, text=True).stdout) for optimized in (False, True)}


@pytest.mark.parametrize('optimized', [False, True])
def test_declaration_admission_also_runs_under_optimized_python(optimized, optimized_admission):
    assert optimized_admission[optimized] == ['TypeError', 'ValueError', *['TypeError'] * 6, 'ValueError', 'ValueError']
