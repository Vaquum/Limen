from pathlib import Path
import json
import subprocess

import polars as pl
import pytest
from click.testing import CliRunner
from ruamel.yaml import YAML

from limen.cli.main import cli
from limen.experiment import UniversalExperimentLoop, _round_storage
from limen.inference import Trainer
from limen.yaml import CompiledSFD, build_search_strategy, parse, validate
from limen.yaml.validator import unknown_errors

ROOT = Path(__file__).resolve().parents[1]
ML = 'limen/yaml/templates/logreg_binary.yaml'
CALIBRATION = 'docs/examples/compiled-research/calibration_assessment.yaml'
CONDITIONAL = 'docs/examples/compiled-research/conditional_features.yaml'
RULE = 'docs/examples/compiled-research/rule_contracts.yaml'
RECOVERY = 'docs/examples/compiled-research/recovery.yaml'
HOLDOUT = 'docs/examples/compiled-research/final/holdout.yaml'
FIXTURE = ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet'


@pytest.mark.parametrize('source,container,key,expected', [
    (ML, (), 'uel_typo', 'uel_typo'),
    (ML, ('sfd',), 'paramz', 'sfd.paramz'),
    (ML, ('metadata',), 'descripton', 'metadata.descripton'),
    (ML, ('sfd', 'manifest'), 'split_date', 'sfd.manifest.split_date'),
    (RULE, ('sfd', 'manifest'), 'features', 'sfd.manifest.features'),
    (ML, ('sfd', 'manifest', 'data_source'), 'parmas', 'sfd.manifest.data_source.parmas'),
    (ML, ('sfd', 'manifest', 'scaler'), 'clas', 'sfd.manifest.scaler.clas'),
    (ML, ('sfd', 'manifest', 'feature_ablation'), 'seed_kye', 'sfd.manifest.feature_ablation.seed_kye'),
    (CALIBRATION, ('sfd', 'manifest', 'calibration'), 'threshold_fn', 'sfd.manifest.calibration.threshold_fn'),
    (CALIBRATION, ('sfd', 'manifest', 'calibration', 'probability_calibration'), 'parmas',
     'sfd.manifest.calibration.probability_calibration.parmas'),
    (CALIBRATION, ('sfd', 'manifest', 'calibration', 'threshold_function'), 'parmas',
     'sfd.manifest.calibration.threshold_function.parmas'),
    (ML, ('sfd', 'manifest', 'indicators', 0), 'parmas', 'sfd.manifest.indicators[0].parmas'),
    (ML, ('sfd', 'manifest', 'pre_split_data_selector'), 'parmas', 'sfd.manifest.pre_split_data_selector.parmas'),
    (ML, ('sfd', 'manifest', 'bar_formation'), 'parmas', 'sfd.manifest.bar_formation.parmas'),
    (CONDITIONAL, ('sfd', 'manifest', 'features', 0), 'include_iff', 'sfd.manifest.features[0].include_iff'),
    (RULE, ('sfd', 'manifest', 'strategy'), 'entyr', 'sfd.manifest.strategy.entyr'),
    (RULE, ('sfd', 'manifest', 'strategy', 'conditions', 3), 'params',
     'sfd.manifest.strategy.conditions[3].params'),
    (RULE, ('sfd', 'manifest', 'strategy', 'conditions', 3), 'lookbak',
     'sfd.manifest.strategy.conditions[3].lookbak'),
    (ML, ('uel',), 'n_permutation', 'uel.n_permutation'),
    (ML, ('uel', 'search_strategy'), 'sed', 'uel.search_strategy.sed'),
    (ML, ('lineage',), 'parnet_id', 'lineage.parnet_id'),
])
def test_unknown_fields_are_single_located_errors_in_fresh_and_saved_admission(
    tmp_path, monkeypatch, source, container, key, expected,
):
    config, errors = parse(ROOT / source)
    assert not errors and validate(config).valid
    if container == ('lineage',):
        config['lineage'] = {}
    if container == ('sfd', 'manifest', 'pre_split_data_selector'):
        config['sfd']['manifest']['pre_split_data_selector'] = {
            'func': 'limen.data.utils.random_slice', 'params': {'rows': 96},
        }
    if container == ('sfd', 'manifest', 'bar_formation'):
        config['sfd']['manifest']['bar_formation'] = {
            'func': 'limen.data.bars.trade_bars', 'params': {'trade_threshold': 100000},
        }
    assert validate(config).valid
    block = config
    for component in container:
        block = block[component]
    block[key] = None
    before = repr(config)
    result = validate(config)
    assert not result.valid
    assert [(error.path, error.message) for error in result.errors] == [(expected, f"Unknown field '{key}'")]
    assert [(error.path, error.message) for error in unknown_errors(config)] == [(expected, f"Unknown field '{key}'")]
    assert repr(config) == before
    (tmp_path / 'limen.toml').write_text('')
    submitted = tmp_path / 'experiment.yaml'
    YAML().dump(config, submitted)
    monkeypatch.chdir(tmp_path)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError('Unknown field reached compilation or source access')

    monkeypatch.setattr(CompiledSFD, '__init__', forbidden)
    monkeypatch.setattr('limen.experiment.manifest_core.DataSourceResolver.resolve', staticmethod(forbidden))
    before = _snapshot(tmp_path)
    invoked = CliRunner().invoke(cli, ['run', '--dry-run', str(submitted)])
    assert invoked.exit_code == 1, invoked.output
    assert f'[{expected}]' in invoked.output
    assert invoked.output.count(f"Unknown field '{key}'") == 1
    assert calls == [] and _snapshot(tmp_path) == before


def _snapshot(directory):
    return {str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob('*') if path.is_file()}


@pytest.mark.parametrize('root,key,message', [
    ((), 'objective', 'Declare the objective only at sfd.manifest.objective'),
    ((), 'acceptance', 'Declare acceptance under sfd.manifest'),
    ((), 'recording', 'Declare recording only under uel'),
    ((), 'feature_shims', 'Declare feature_shims only under sfd.manifest'),
    (('sfd',), 'recording', 'Declare recording only under uel'),
    (('sfd',), 'feature_shims', 'Declare feature_shims only under sfd.manifest'),
    (('uel',), 'objective', 'Declare the objective only at sfd.manifest.objective'),
    (('uel',), 'acceptance', 'Declare acceptance under sfd.manifest'),
    (('uel',), 'feature_shims', 'Declare feature_shims only under sfd.manifest'),
])
def test_specific_misplacement_diagnostics_keep_one_owner(root, key, message):
    config, errors = parse(ROOT / ML)
    assert not errors
    block = config
    for component in root:
        block = block[component]
    block[key] = {}
    expected = '.'.join((*root, key))
    for diagnosed in (validate(config).errors, unknown_errors(config)):
        assert [(error.path, error.message) for error in diagnosed] == [(expected, message)]


def test_root_typo_does_not_hide_other_supported_schema_errors():
    config, errors = parse(ROOT / ML)
    assert not errors
    config['uel_typo'] = {}
    config['sfd']['manifest']['backtest'] = {'fee_bp': 1}
    result = validate(config)
    assert {error.path for error in result.errors} == {'uel_typo', 'sfd.manifest.backtest.fee_bp'}


def test_recorded_commit_uri_fork_recommit_and_native_readback_remain_valid(tmp_path, monkeypatch):
    from limen.yaml.store import resolve_manifest_uri

    monkeypatch.chdir(tmp_path)
    (tmp_path / 'limen.toml').write_text('')
    for arguments in (['init', '--quiet'], ['config', 'user.email', 'test@example.com'], ['config', 'user.name', 'Test']):
        subprocess.run(['git', *arguments], cwd=tmp_path, check=True)
    config, errors = parse(ROOT / RECOVERY)
    assert not errors
    config['metadata']['mode'] = 'production'
    config['sfd']['manifest']['data_source']['params']['file_path_or_url'] = str(FIXTURE)
    config['uel']['n_permutations'] = 2
    submitted = tmp_path / 'experiment.yaml'
    YAML().dump(config, submitted)
    committed = CliRunner().invoke(cli, ['commit', str(submitted)])
    assert committed.exit_code == 0, committed.output
    stored = list((tmp_path / 'manifests/committed').glob('*.yaml'))
    assert len(stored) == 1
    identity = 'sha256:' + stored[0].stem
    restored, errors = parse(stored[0])
    assert not errors and validate(restored).valid and not unknown_errors(restored)
    assert restored['lineage']['id'] == identity and 'committed_at' in restored['lineage']
    run = CliRunner().invoke(cli, ['run', '--no-progress-bar', 'manifest://' + identity])
    assert run.exit_code == 0, run.output
    outputs = [path.parent for path in (tmp_path / 'results').rglob('metadata.json')]
    assert len(outputs) == 1
    assert pl.read_csv(outputs[0] / 'results.csv').height == 2
    with _round_storage.open_text(_round_storage.round_path(outputs[0])) as stream:
        rounds = [json.loads(line) for line in stream if line.strip()]
    before = _snapshot(tmp_path)
    sensor = Trainer(outputs[0], data=pl.read_parquet(FIXTURE)).train([rounds[0]['round_id']])[0]
    assert sensor is not None and _snapshot(tmp_path) == before
    fork = CliRunner().invoke(cli, ['fork', identity, 'structural_child'])
    assert fork.exit_code == 0, fork.output
    child = tmp_path / 'manifests/structural_child.yaml'
    document, errors = parse(child)
    assert not errors and document['lineage'] == {'parent_id': identity}
    assert validate(document).valid and not unknown_errors(document)
    document['metadata']['mode'] = 'production'
    YAML().dump(document, child)
    recommitted = CliRunner().invoke(cli, ['commit', str(child)])
    assert recommitted.exit_code == 0, recommitted.output
    children = [path for path in (tmp_path / 'manifests/committed').glob('*.yaml') if path != stored[0]]
    assert len(children) == 1
    child_id = 'sha256:' + children[0].stem
    verified, root = resolve_manifest_uri('manifest://' + child_id, tmp_path)
    readback, errors = parse(verified)
    assert root == tmp_path and not errors and validate(readback).valid
    assert readback['lineage']['id'] == child_id
    assert readback['lineage']['parent_id'] == identity


@pytest.mark.parametrize('unknown', (False, True))
def test_genuine_saved_checkpoint_unknown_admission_preserves_evidence(tmp_path, monkeypatch, unknown):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'limen.toml').write_text('')
    config, errors = parse(ROOT / RECOVERY)
    assert not errors
    config['sfd']['manifest']['data_source']['params']['file_path_or_url'] = str(FIXTURE)
    if unknown:
        config['uel_typo'] = {}
    raw = pl.read_parquet(FIXTURE)
    directory = tmp_path / 'saved'

    def stop(log, _msq):
        if log.height == 2:
            loop._shutdown_requested = True

    # The native prevalidated API produces real legacy evidence without rewriting it.
    loop = UniversalExperimentLoop(sfd=CompiledSFD(config), data=raw, experiment_dir=directory,
        search_strategy=build_search_strategy(config), yaml_reference=config,
        feedback_interval=1, checkpoint_interval=1, intra_callback=stop)
    loop.run('legacy_structural', n_permutations=4, prep_each_round=True, progress_bar=False,
             recording=config['uel']['recording'])
    assert loop.interrupted and pl.read_csv(directory / 'results.csv').height == 2
    metadata = json.loads((directory / 'metadata.json').read_text())
    assert ('uel_typo' in metadata['yaml_reference']) is unknown
    with _round_storage.open_text(_round_storage.round_path(directory)) as stream:
        rounds = [json.loads(line) for line in stream if line.strip()]
    before = _snapshot(tmp_path)
    assert Trainer(directory, data=raw).train([rounds[0]['round_id']])[0] is not None
    assert _snapshot(tmp_path) == before
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError('Unknown saved field reached checkpoint, compilation or source')

    with monkeypatch.context() as guard:
        if unknown:
            guard.setattr(CompiledSFD, '__init__', forbidden)
            guard.setattr('limen.experiment.checkpoint_manager.CheckpointManager.load', forbidden)
            guard.setattr('limen.experiment.manifest_core.DataSourceResolver.resolve', staticmethod(forbidden))
        result = CliRunner().invoke(cli, ['run', '--resume', str(directory), '--no-progress-bar'])
    if unknown:
        assert result.exit_code == 1 and "Unknown field 'uel_typo'" in result.output
        assert calls == [] and _snapshot(tmp_path) == before
    else:
        assert result.exit_code == 0, result.output
        assert pl.read_csv(directory / 'results.csv').height == 4


def test_legacy_stored_unknown_uri_refuses_without_rewriting_store(tmp_path, monkeypatch):
    from limen.yaml.store import commit_manifest

    monkeypatch.chdir(tmp_path)
    (tmp_path / 'limen.toml').write_text('')
    config, errors = parse(ROOT / RECOVERY)
    assert not errors
    config['metadata']['mode'] = 'production'
    config['uel_typo'] = {}
    submitted = tmp_path / 'legacy.yaml'
    YAML().dump(config, submitted)
    identity, _ = commit_manifest(submitted, tmp_path)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError('Unknown stored field reached compilation or source')

    monkeypatch.setattr(CompiledSFD, '__init__', forbidden)
    monkeypatch.setattr('limen.experiment.manifest_core.DataSourceResolver.resolve', staticmethod(forbidden))
    before = _snapshot(tmp_path)
    for arguments in (['run'], ['run', '--dry-run']):
        result = CliRunner().invoke(cli, [*arguments, 'manifest://' + identity])
        assert result.exit_code == 1 and "Unknown field 'uel_typo'" in result.output
        assert calls == [] and _snapshot(tmp_path) == before


@pytest.mark.parametrize('source,container,field,typo,path', [
    (CONDITIONAL, ('sfd', 'manifest', 'features', 0), 'include_if', 'include_iff', 'sfd.manifest.features[0].include_iff'),
    (RULE, ('sfd', 'manifest', 'strategy', 'conditions', 3), 'lookback', 'lookbak', 'sfd.manifest.strategy.conditions[3].lookbak'),
    (HOLDOUT, (), 'holdout', 'holduot', 'holduot'),
])
def test_optional_control_replacements_refuse_all_fresh_cli_boundaries(
    tmp_path, monkeypatch, source, container, field, typo, path,
):
    config, errors = parse(ROOT / source)
    assert not errors and validate(config).valid
    block = config
    for component in container:
        block = block[component]
    block[typo] = block.pop(field)
    assert [error.path for error in validate(config).errors] == [path]
    assert [error.path for error in unknown_errors(config)] == [path]
    config['metadata']['mode'] = 'production'
    submitted = tmp_path / 'optional.yaml'
    YAML().dump(config, submitted)
    (tmp_path / 'limen.toml').write_text('')
    monkeypatch.chdir(tmp_path)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError('Misspelled optional control reached construction or source')

    monkeypatch.setattr(CompiledSFD, '__init__', forbidden)
    monkeypatch.setattr('limen.experiment.manifest_core.DataSourceResolver.resolve', staticmethod(forbidden))
    before = _snapshot(tmp_path)
    for command in (['validate'], ['run', '--dry-run'], ['run', '--no-progress-bar'], ['profile'], ['commit']):
        result = CliRunner().invoke(cli, [*command, str(submitted)])
        assert result.exit_code == 1, result.output
        assert f'[{path}]' in result.output and result.output.count(f"Unknown field '{typo}'") == 1
        assert calls == [] and _snapshot(tmp_path) == before
