import copy
import json
from pathlib import Path
import shutil

import polars as pl
import pytest
from click.testing import CliRunner
from ruamel.yaml import YAML
from threadpoolctl import threadpool_limits

from limen.cli.main import cli
from limen.cohort import Cohort
from limen.data.bars import trade_bars
from limen.experiment import UniversalExperimentLoop, _round_storage
from limen.experiment.manifest_core import DataSourceResolver
from limen.inference import Trainer
from limen.targets import NextReturnTarget
from limen.yaml import CompiledSFD, build_manifest, parse, validate
from docs.examples.tests.helper_oracles import verify_rules

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet'


@pytest.fixture(autouse=True)
def single_thread():
    with threadpool_limits(limits=1):
        yield


def _config():
    config, errors = parse(ROOT / 'docs/examples/compiled-research/rule_contracts.yaml')
    assert not errors
    config['sfd']['manifest']['data_source']['params']['file_path_or_url'] = str(FIXTURE)
    return config


@pytest.mark.parametrize('field', ['target', 'bar_formation'])
@pytest.mark.parametrize('value', [None, {}, False, 0, '', [], {'func': 'limen.does.not.exist'}])
def test_rule_yaml_rejects_discarded_fields_by_presence(field, value):
    config = _config()
    config['sfd']['manifest'][field] = value
    result = validate(config)
    path = f'sfd.manifest.{field}'
    assert not result.valid
    assert [(error.path, error.message) for error in result.errors] == [(path, f"Field '{field}' is unsupported in rule_based YAML and would be discarded")]
    assert not [warning for warning in result.warnings if warning.path == path]
    for compile_manifest in (lambda: build_manifest(config), lambda: CompiledSFD(config).manifest()):
        with pytest.raises(ValueError, match=rf'{path}:.*unsupported in rule_based YAML.*discarded'):
            compile_manifest()


def _declare(config, field):
    if field == 'target':
        config['sfd']['manifest'][field] = {'name': 'next_return', 'class': 'limen.targets.NextReturnTarget',
            'transform_params': {'periods': 3, 'scale': 100.0}}
    else:
        config['sfd']['manifest'][field] = {'func': 'limen.data.bars.trade_bars',
            'params': {'trade_threshold': '{trade_threshold}'}}
        config['sfd']['params']['trade_threshold'] = [100000, 200000]


def _forbidden(*args, **kwargs):
    raise AssertionError('Rejected rule YAML reached compilation or source fetch')


def _snapshot(directory):
    return {str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob('*') if path.is_file()}


def _records(directory):
    return [json.loads(line) for line in _round_storage.text(_round_storage.round_path(directory)).splitlines()]


@pytest.mark.parametrize('field', ['target', 'bar_formation'])
@pytest.mark.parametrize('command', [('validate',), ('run', '--dry-run'), ('run', '--no-progress-bar'), ('commit',)])
def test_public_rule_admission_precedes_compilation_and_writes(tmp_path, monkeypatch, field, command):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'limen.toml').write_text('')
    config = _config()
    _declare(config, field)
    if command[0] == 'commit':
        config['metadata']['mode'] = 'production'
    path = tmp_path / 'rejected.yaml'
    YAML().dump(config, path)
    before = _snapshot(tmp_path)
    monkeypatch.setattr(CompiledSFD, 'manifest', _forbidden)
    monkeypatch.setattr(DataSourceResolver, 'resolve', staticmethod(_forbidden))
    result = CliRunner().invoke(cli, [*command, str(path)])
    assert result.exit_code == 1, result.output
    assert f'[sfd.manifest.{field}]' in result.output
    assert 'unsupported in rule_based YAML and would be discarded' in result.output
    assert _snapshot(tmp_path) == before
    assert not (tmp_path / 'results').exists()
    assert not (tmp_path / 'manifests').exists()


def test_rule_rejections_are_ordered_and_other_unknown_fields_are_errors():
    config = _config()
    for field in ('target', 'bar_formation'):
        _declare(config, field)
    result = validate(config)
    assert [error.path for error in result.errors] == ['sfd.manifest.target', 'sfd.manifest.bar_formation']
    assert not result.warnings
    with pytest.raises(ValueError, match=r'sfd.manifest.target:.*sfd.manifest.bar_formation:'):
        build_manifest(config)
    config = _config()
    config['sfd']['manifest'].update(features=[], strict_mode=False,
        pre_split_data_selector={'func': 'limen.data.selectors.random_slice', 'params': {'n_rows': 100}}, unrelated_rule_field=True)
    result = validate(config)
    assert not result.valid and not result.warnings
    assert {error.path for error in result.errors} == {f'sfd.manifest.{field}' for field in
        ('features', 'strict_mode', 'pre_split_data_selector', 'unrelated_rule_field')}


@pytest.mark.parametrize('field', [None, 'target', 'bar_formation'])
def test_saved_rule_reconstruction_and_real_checkpoint_resume(tmp_path, monkeypatch, field):
    import limen.cli.commands._load_yaml as loader
    from limen.yaml import compiler

    monkeypatch.chdir(tmp_path)
    config = _config()
    config['uel'].update(output_path='rule_checkpoint', checkpoint_interval=1)
    if field is not None:
        _declare(config, field)
        config['sfd']['params'] = {'price_floor': [40000., 41000.]}  # Keep a two-round legacy campaign.
        if field == 'bar_formation':
            config['sfd']['manifest'][field]['params']['trade_threshold'] = 100000
    path = tmp_path / 'legacy.yaml'
    YAML().dump(config, path)
    original_append = UniversalExperimentLoop._append_round_data

    def stop_after_record(loop, *args, **kwargs):
        original_append(loop, *args, **kwargs)
        loop._shutdown_requested = True

    def legacy_validation(document):
        admitted = copy.deepcopy(document)
        for key in ('target', 'bar_formation'):
            admitted['sfd']['manifest'].pop(key, None)
        return validate(admitted)

    # Produce the actual old compiler's run/checkpoint, without editing saved evidence.
    with monkeypatch.context() as legacy:
        legacy.setattr(UniversalExperimentLoop, '_append_round_data', stop_after_record)
        if field is not None:
            legacy.setattr(loader, '_validate', legacy_validation)
            legacy.setattr(compiler, 'RULE_YAML_REJECTIONS', {})
        result = CliRunner().invoke(cli, ['run', '--no-progress-bar', str(path)])
    assert result.exit_code == 1 and 'Interrupted; checkpoint retained' in result.output
    directory = tmp_path / 'results/dev/rule_checkpoint'
    assert len(_records(directory)) == 1
    metadata = json.loads((directory / 'metadata.json').read_text())
    if field is not None:
        assert field in metadata['yaml_reference']['sfd']['manifest']
    before = _snapshot(directory)
    raw = pl.read_parquet(FIXTURE)
    if field is not None:
        with monkeypatch.context() as guard:
            guard.setattr(DataSourceResolver, 'resolve', staticmethod(_forbidden))
            result = CliRunner().invoke(cli, ['run', '--no-progress-bar', '--resume', str(directory)])
            assert result.exit_code == 1 and f'sfd.manifest.{field}' in result.output
            assert 'unsupported in rule_based YAML' in result.output
            with pytest.raises(ValueError, match='not a valid YAML SFD reference') as trainer_error:
                Trainer(directory, data=raw)
            assert f'sfd.manifest.{field}:' in str(trainer_error.value.__cause__)
            with pytest.raises(ValueError, match=rf'sfd.manifest.{field}:.*unsupported in rule_based YAML'):
                Cohort(rounds=[(directory, _records(directory)[0]['round_id'])])
        assert _snapshot(directory) == before
    else:
        result = CliRunner().invoke(cli, ['run', '--no-progress-bar', '--resume', str(directory)])
        assert result.exit_code == 0, result.output
        records = _records(directory)
        assert len(records) == 2
        sensors = Trainer(directory, data=raw).train([record['round_id'] for record in records])
        assert len(sensors) == 2
        assert len(Cohort(rounds=[(directory, record['round_id']) for record in records]).member_ids) == 2
        verify_rules({'assessment_rows': 96, 'history_resets_at_split': True, 'rows': [
            {'price_floor': record['round_params']['price_floor'], 'signals': record['preds'],
             'positive_decisions': sum(record['preds'])} for record in records]}, FIXTURE.parent)
        assert all(len(record['preds']) == 96 and sum(record['preds']) == 32 for record in records)


@pytest.mark.parametrize('kind,expected', [('base', 96), ('target', 93), ('formation', 35)])
def test_native_rule_target_and_active_formation_remain_supported(kind, expected):
    config = _config()
    assert validate(config).valid and not validate(config).warnings
    manifest = CompiledSFD(config).manifest()
    params = {'price_floor': 40000.}
    if kind == 'target':
        manifest.with_target_label('next_return', NextReturnTarget, transform_params={'periods': 3, 'scale': 100.0})
    elif kind == 'formation':
        manifest.set_bar_formation(trade_bars, trade_threshold=100000)
        params['bar_type'] = 'trade'
    data = manifest.prepare_data(pl.read_parquet(FIXTURE), params)
    result = manifest.run_model(data, params)
    assert len(result['_preds']) == expected


def test_recorded_ml_yaml_keeps_target_and_all_active_bar_formations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(ROOT / 'docs/examples/tests'))
    from cli_proof import verify_bars

    config, errors = parse(ROOT / 'docs/examples/compiled-research/bars.yaml')
    assert not errors and validate(config).valid
    (tmp_path / 'inputs').mkdir()
    name = 'spot_15m_20250101_20250531.parquet'
    shutil.copy2(ROOT / 'tests/fixtures' / name, tmp_path / 'inputs' / name)
    path = tmp_path / 'bars.yaml'
    YAML().dump(config, path)
    manifest = CompiledSFD(config).manifest()
    assert manifest.target_class_config is not None and manifest.bar_formation is not None
    result = CliRunner().invoke(cli, ['run', '--no-progress-bar', str(path)])
    assert result.exit_code == 0, result.output
    proof = verify_bars(tmp_path)
    assert len(proof['populations']) == 9
