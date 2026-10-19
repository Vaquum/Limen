import copy
import csv
import gzip
import json
import shutil
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from math import prod
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from ruamel.yaml import YAML

from limen.experiment import _round_storage as storage
from limen.experiment import RuleBasedManifest
from limen.experiment.acceptance_report import acceptance_report
from limen.experiment.errors import StrictModeError
from limen.experiment.family_acceptance_report import family_acceptance_report
from limen.cli.commands.run import run_experiment
from limen.metrics import deflated_sharpe_ratio, probability_of_backtest_overfitting
from limen.yaml.store import STORE_RELATIVE, commit_manifest
from tests.test_walk_forward_uel import _bars, _config, _loop, _run


def _experiment(config, bars, directory, project, *, parent=None):
    manifest = project / 'experiment.yaml'
    with manifest.open('w') as stream:
        YAML().dump(config, stream)
    identity, _ = commit_manifest(manifest, project, parent_id=parent)
    loop = _loop(config, bars, directory)
    _run(loop, n_permutations=prod(len(values) for values in config['sfd']['params'].values()))
    return identity


def _tracks(directory):
    frame = pl.read_parquet(directory / 'trial_returns.parquet')
    return {trial: frame.filter(pl.col('trial') == trial).sort('bar')['net_return'].to_numpy()
            for trial in frame['trial'].unique(maintain_order=True).to_list()}


def _copy_run(source, target):
    shutil.copytree(source, target)
    return target


def _records(directory):
    with storage.open_text(storage.round_path(directory)) as lines:
        return [json.loads(line) for line in lines]


def _write_records(directory, records):
    path = storage.round_path(directory)
    path.unlink()
    for record in records:
        storage.append_record(path, record)


def _files(*directories):
    return {str(path): (sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for directory in directories for path in directory.rglob('*') if path.is_file()}


@pytest.fixture(scope='module')
def family_runs(tmp_path_factory):
    project = tmp_path_factory.mktemp('recorded_family')
    parent, child, repeated, varied = [project / name for name in ('01-parent', '02-child', '03-repeat', '04-features')]
    config = _config()
    config['metadata']['name'] = 'family_parent'
    root_id = _experiment(config, _bars(), parent, project)
    child_config = copy.deepcopy(config)
    child_config['metadata']['name'] = 'family_child'
    child_config['sfd']['params']['entry_return'] = [0.001, 0.002]
    child_id = _experiment(child_config, _bars(), child, project, parent=root_id)
    _copy_run(child, repeated)
    varied_config = copy.deepcopy(config)
    varied_config['metadata']['name'] = 'family_features'
    varied_config['sfd']['manifest']['indicators'].append({
        'func': 'limen.indicators.body_pct', 'params': {},
    })
    varied_config['sfd']['params']['entry_return'] = [0.0]
    _experiment(varied_config, _bars(), varied, project, parent=root_id)
    return project, [parent, child, repeated, varied], root_id, child_id


def test_family_combines_comparable_runs(family_runs):
    project, paths, root_id, child_id = family_runs
    report = family_acceptance_report(list(reversed(paths)), project_root=project)
    assert report == family_acceptance_report(paths, project_root=project)
    assert len(report['comparisons']) == 1
    group = report['comparisons'][0]
    parent_tracks = _tracks(paths[0])
    tracks = [track for path in paths for track in _tracks(path).values()]
    matrix = np.stack(tracks)
    sharpes = matrix.mean(axis=1) / matrix.std(axis=1, ddof=1)
    variance = float(sharpes.var(ddof=1))
    winner = int(np.argmax(sharpes))
    assert group['n_trials'] == group['n_family_trials'] == 7
    assert group['n_trials_outside_group'] == group['n_trials_without_group'] == 0
    assert group['n_bars'] == matrix.shape[1]
    assert group['winner_sharpe'] == pytest.approx(float(sharpes[winner]))
    assert group['trial_sharpe_variance'] == pytest.approx(variance)
    assert group['deflated_sharpe_probability'] == pytest.approx(
        deflated_sharpe_ratio(matrix[winner], n_trials=7, trial_sharpe_variance=variance),
    )
    assert group['pbo'] is None
    assert list(group['errors']) == ['pbo']
    assert 'open/high/low' in group['errors']['pbo']
    assert group['members'][0] == {'directory': str(paths[0]), 'trial': next(iter(parent_tracks))}
    child_run = next(run for run in report['runs'] if run['directory'] == str(paths[1]))
    assert [item['manifest_id'] for item in child_run['lineage']] == [root_id, child_id]
    assert all(item['available'] for item in child_run['lineage'])
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert not (project / 'acceptance_report.json').exists()


def test_family_preserves_variations_and_identifies_repeats(family_runs):
    project, paths, _, _ = family_runs
    report = family_acceptance_report([*paths, paths[0]], project_root=project)
    assert report['coverage'] == {
        'recorded_trials': 7, 'distinct_compared_trials': 7, 'repeats': 0, 'evidence_matches': 3,
        'unavailable': 0, 'incompatible': 0, 'known_failed_trials': 0, 'runs_without_terminal_inventory': 0,
    }
    assert report['duplicate_inputs'] == [str(paths[0])]
    matches = [trial for trial in report['trials'] if trial['evidence_match_of'] is not None]
    assert len(matches) == 3
    assert all(trial['repeat_identity'] == 'unproven' and trial['comparison'] for trial in matches)
    assert all(trial['evidence_match_of']['directory'] != str(paths[2]) for trial in matches)
    assert all(trial['repeat_of'] is None for trial in report['trials'])
    original = next(trial for trial in report['trials'] if trial['directory'] == str(paths[0]))
    varied = next(trial for trial in report['trials'] if trial['directory'] == str(paths[3]))
    np.testing.assert_array_equal(_tracks(paths[0])[original['trial']], _tracks(paths[3])[varied['trial']])
    assert original['trial'] == varied['trial']
    assert original['candidate'] != varied['candidate']
    assert original['comparison'] == varied['comparison']
    assert varied['repeat_of'] is None


def test_family_explains_comparability_and_provenance(family_runs, tmp_path):
    project, paths, _, _ = family_runs
    missing = _copy_run(paths[0], tmp_path / 'missing_folds')
    with storage.open_text(storage.round_path(missing)) as lines:
        records = [json.loads(line) for line in lines]
    storage.round_path(missing).unlink()
    for record in records:
        del record['folds']
        storage.append_record(storage.round_path(missing), record)
    shifted = tmp_path / 'shifted'
    config = _config()
    config['metadata']['name'] = 'family_shifted'
    fixture = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'
    _experiment(config, pl.read_parquet(fixture).slice(1, 384), shifted, project)
    report = family_acceptance_report([missing, shifted, paths[0]], project_root=project)
    assert len(report['comparisons']) == 2
    assert report['coverage']['unavailable'] == 2
    unavailable = [trial for trial in report['trials'] if trial['unavailable']]
    assert all(trial['directory'] == str(missing) and 'fold evidence is missing' in trial['unavailable'] for trial in unavailable)
    first, second = report['comparisons']
    assert first['evidence']['folds'][0]['first_test_datetime'] != second['evidence']['folds'][0]['first_test_datetime']
    assert first['n_trials'] == second['n_trials'] == 2
    assert first['n_bars'] == second['n_bars'] == 94
    assert 'No sample intersection' in report['comparability']
    absent_store = family_acceptance_report([paths[1]], project_root=tmp_path)
    assert absent_store['runs'][0]['lineage'][0]['available'] is False
    assert 'Recorded manifest ancestry is incomplete' in absent_store['runs'][0]['errors']


@pytest.fixture(scope='module')
def signed_run(family_runs, tmp_path_factory):
    project, _, _, _ = family_runs
    config = _config()
    config['metadata']['name'] = 'family_signed'
    config['sfd']['manifest']['backtest']['product'] = {
        'kind': 'cash_spot', 'instrument': 'BTCUSDT', 'base_currency': 'BTC',
        'quote_currency': 'USDT', 'quantity_step': 1e-9, 'min_notional': 0.0,
    }
    signed = tmp_path_factory.mktemp('signed_family') / 'run'
    _experiment(config, _bars(), signed, project)
    return signed


def test_family_separates_signed_execution(family_runs, signed_run):
    project, paths, _, _ = family_runs
    signed = signed_run
    report = family_acceptance_report([paths[0], signed], project_root=project)
    assert len(report['comparisons']) == 2
    assert {group['evidence']['folds'][0]['return_semantics'] for group in report['comparisons']} == {'snapshot', 'signed_execution'}
    assert all(group['n_trials'] == 2 for group in report['comparisons'])
    signed_group = next(group for group in report['comparisons'] if group['evidence']['folds'][0]['return_semantics'] == 'signed_execution')
    assert signed_group['evidence']['folds'][0]['product']['quote_currency'] == 'USDT'
    assert signed_group['pbo'] == probability_of_backtest_overfitting(np.stack(list(_tracks(signed).values())), n_blocks=2)
    assert all(group['deflated_sharpe_probability'] is None for group in report['comparisons'])


def test_family_supported_single_run_matches_per_run(family_runs, signed_run, tmp_path):
    project, _, _, _ = family_runs
    before = _files(signed_run, project / STORE_RELATIVE)
    oracle = acceptance_report(_copy_run(signed_run, tmp_path / 'oracle'))
    group = family_acceptance_report([signed_run], project_root=project)['comparisons'][0]
    assert group['n_trials'] == oracle['n_trials']
    assert group['winner']['trial'] == oracle['winner_trial']
    for key in ('winner_sharpe', 'trial_sharpe_variance', 'deflated_sharpe_probability', 'pbo'):
        assert group[key] == oracle[key]
    assert _files(signed_run, project / STORE_RELATIVE) == before


@pytest.mark.parametrize('source_params', (None, {}))
def test_family_accepts_empty_source_params(family_runs, tmp_path, source_params):
    project, _, _, _ = family_runs
    config = _config()
    config['metadata']['name'] = 'family_empty_source_params'
    config['sfd']['manifest']['data_source']['params'] = source_params
    directory = tmp_path / 'empty_params'
    _experiment(config, _bars(), directory, project)
    report = family_acceptance_report([directory], project_root=project)
    assert report['coverage']['unavailable'] == 0
    assert report['comparisons'][0]['evidence']['source']['params'] == {}
    assert report['comparisons'][0]['n_trials'] == 2


def test_family_rejects_corruption_and_explains_metric_limits(family_runs, signed_run, tmp_path):
    project, paths, _, _ = family_runs
    damaged = _copy_run(paths[0], tmp_path / 'duplicate_bar')
    frame = pl.read_parquet(damaged / 'trial_returns.parquet')
    pl.concat([frame, frame.head(1)]).write_parquet(damaged / 'trial_returns.parquet')
    with pytest.raises(ValueError, match='unique contiguous'):
        family_acceptance_report([damaged], project_root=project)
    missing = tmp_path / 'missing_returns'
    missing.mkdir()
    with pytest.raises(FileNotFoundError, match='metadata.json'):
        family_acceptance_report([missing], project_root=project)
    shutil.copy2(paths[0] / 'metadata.json', missing / 'metadata.json')
    report = family_acceptance_report([missing], project_root=project)
    assert report['comparisons'] == []
    assert any('coverage is unknown' in error for error in report['runs'][0]['errors'])
    blocked = family_acceptance_report([signed_run], project_root=project, n_blocks=4)['comparisons'][0]
    assert blocked['pbo'] is None
    assert 'equal contiguous blocks' in blocked['errors']['pbo']
    assert isinstance(blocked['deflated_sharpe_probability'], float)
    for count in (1, 3, True, 2.0):
        with pytest.raises(ValueError, match='even integer'):
            family_acceptance_report(paths, project_root=project, n_blocks=count)
    with pytest.raises(ValueError, match='at least one'):
        family_acceptance_report([], project_root=project)


def test_family_keeps_unproven_training_and_source_variations(family_runs, tmp_path):
    project, paths, _, _ = family_runs
    config = _config()
    different_training = tmp_path / 'different_training'
    _experiment(config, _bars().tail(383), different_training, project)
    config['sfd']['manifest']['data_source']['params']['row_count_limit'] = 336
    different_declaration = tmp_path / 'different_declaration'
    _experiment(config, _bars(), different_declaration, project)
    unknown = _copy_run(paths[0], tmp_path / 'unknown_implementation')
    metadata = json.loads((unknown / 'metadata.json').read_text())
    metadata['limen_version'] = 'dev'
    (unknown / 'metadata.json').write_text(json.dumps(metadata))
    report = family_acceptance_report([paths[0], different_training, different_declaration, unknown], project_root=project)
    assert len(report['comparisons']) == 1
    assert report['comparisons'][0]['n_trials'] == 8
    assert all(trial['repeat_of'] is None and trial['repeat_identity'] == 'unproven' for trial in report['trials'])
    assert all(trial['evidence']['source_verified'] is False for trial in report['trials'])


def test_family_refuses_fragmented_and_singleton_deflation(family_runs, tmp_path):
    project, paths, _, _ = family_runs
    config = _config()
    config['sfd']['manifest']['indicators'].append({
        'func': 'limen.indicators.window_return', 'params': {'period': '{lookback}'},
    })
    config['sfd']['params']['lookback'] = [2, 6, 12]
    fragmented = tmp_path / 'lookbacks'
    _experiment(config, _bars(), fragmented, project)
    report = family_acceptance_report([fragmented], project_root=project)
    assert [(group['n_trials'], group['n_bars']) for group in report['comparisons']] == [(2, 92), (2, 84), (2, 72)]
    assert acceptance_report(_copy_run(fragmented, tmp_path / 'oracle'))['deflated_sharpe_probability'] is None
    assert all(group['n_family_trials'] == 6 and group['n_trials_outside_group'] == 4 for group in report['comparisons'])
    assert all(group['deflated_sharpe_probability'] is None and 'incomplete' in group['errors']['deflated_sharpe_probability'] for group in report['comparisons'])
    across_runs = family_acceptance_report([paths[0], fragmented], project_root=project)
    assert len(across_runs['comparisons']) == 4
    assert all(group['deflated_sharpe_probability'] is None for group in across_runs['comparisons'])
    single = copy.deepcopy(_config())
    single['sfd']['params']['entry_return'] = [0.0]
    directory = tmp_path / 'singleton'
    _experiment(single, _bars(), directory, project)
    group = family_acceptance_report([directory], project_root=project)['comparisons'][0]
    assert group['deflated_sharpe_probability'] is None
    assert 'singleton' in group['errors']['deflated_sharpe_probability']


@pytest.mark.parametrize('profile', ('minimum', 'lean', 'without_execution'))
def test_documented_recording_profile_without_helper_overrides(tmp_path, profile):
    config = _config()
    config['metadata'].update(name='family_' + profile, mode='production')
    config['sfd']['manifest']['data_source'] = {
        'method': 'limen.data.HistoricalData.get_any_file',
        'params': {'file_path_or_url': str(Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'), 'row_count_limit': 384},
    }
    if profile != 'lean':
        config['uel'].update(recording={'alignment': True, 'fold_details': True}, record_execution=profile == 'minimum')
    manifest = tmp_path / 'manifest.yaml'
    with manifest.open('w') as stream:
        YAML().dump(config, stream)
    assert run_experiment(manifest, results_base=tmp_path, progress_bar=False)
    directories = [path.parent for path in (tmp_path / 'results').rglob('metadata.json')]
    assert len(directories) == 1
    report = family_acceptance_report(directories, project_root=tmp_path)
    assert report['coverage']['recorded_trials'] == 2
    if profile == 'minimum':
        assert len(report['comparisons']) == 1
        assert isinstance(report['comparisons'][0]['deflated_sharpe_probability'], float)
        assert 'open/high/low' in report['comparisons'][0]['errors']['pbo']
    else:
        assert report['coverage']['unavailable'] == 2
        assert all(trial['recording']['record_execution'] is not True for trial in report['trials'])


def test_family_retains_flat_candidates(family_runs, tmp_path):
    project, _, _, _ = family_runs
    config = _config()
    config['sfd']['params']['entry_return'] = [0.0, 0.5]
    path = tmp_path / 'flat_candidate'
    _experiment(config, _bars(), path, project)
    group = family_acceptance_report([path], project_root=project)['comparisons'][0]
    flat = next(trial for trial, track in _tracks(path).items() if np.std(track) == 0)
    assert group['n_trials'] == 2 and group['deflated_sharpe_probability'] is None
    assert flat in group['errors']['deflated_sharpe_probability']


def test_family_signed_cost_variations_and_recording_omissions(family_runs, signed_run, tmp_path):
    project, _, _, _ = family_runs
    metadata = json.loads((signed_run / 'metadata.json').read_text())
    config = metadata['yaml_reference']
    config['sfd']['manifest']['backtest']['fee_bps'] = '{fee}'
    config['sfd']['params']['fee'] = [7.0, 9.0]
    varied = tmp_path / 'signed_costs'
    _experiment(config, _bars(), varied, project)
    report = family_acceptance_report([signed_run, varied], project_root=project)
    assert len(report['comparisons']) == 1
    group = report['comparisons'][0]
    assert group['n_trials'] == 6
    assert group['pbo'] == probability_of_backtest_overfitting(np.stack([*list(_tracks(signed_run).values()), *list(_tracks(varied).values())]), n_blocks=2)
    projected = _copy_run(signed_run, tmp_path / 'projected')
    records = _records(projected)
    for record in records:
        for fold in record['folds']:
            del fold['trade_contract']['signal_rows']
    _write_records(projected, records)
    omitted = family_acceptance_report([projected, signed_run], project_root=project)
    assert omitted['coverage']['unavailable'] == 2
    assert all('signal availability' in trial['unavailable'] for trial in omitted['trials'] if trial['directory'] == str(projected))


def test_family_report_order_and_timestamp_dispositions(family_runs, tmp_path):
    project, paths, _, _ = family_runs
    unknown = _copy_run(paths[0], tmp_path / 'unknown_start')
    metadata = json.loads((unknown / 'metadata.json').read_text())
    del metadata['created_at']
    (unknown / 'metadata.json').write_text(json.dumps(metadata))
    report = family_acceptance_report([unknown, paths[0]], project_root=project)
    assert report == family_acceptance_report([paths[0], unknown], project_root=project)
    assert report['runs'][-1]['directory'] == str(unknown) and report['runs'][-1]['created_at'] is None
    first, second = [_copy_run(paths[0], tmp_path / name) for name in ('utc', 'offset')]
    for directory, zone in ((first, timezone.utc), (second, timezone(timedelta(hours=2)))):
        records = _records(directory)
        for record in records:
            for fold in record['folds']:
                dates = fold['alignment']['test_datetimes']
                fold['alignment']['test_datetimes'] = [datetime.fromisoformat(value).replace(tzinfo=timezone.utc).astimezone(zone).isoformat() for value in dates]
        _write_records(directory, records)
    equivalent = family_acceptance_report([first, second], project_root=project)
    assert len(equivalent['comparisons']) == 1
    assert equivalent == family_acceptance_report([second, first], project_root=project)
    records = _records(second)
    dates = records[0]['folds'][0]['alignment']['test_datetimes']
    dates[0] = datetime.fromisoformat(dates[0]).replace(tzinfo=None).isoformat()
    _write_records(second, records)
    incompatible = family_acceptance_report([first, second], project_root=project)
    assert incompatible['coverage']['incompatible'] == 1
    assert incompatible['comparisons'][0]['n_trials_without_group'] == 1
    assert incompatible['comparisons'][0]['deflated_sharpe_probability'] is None
    assert json.loads(json.dumps(incompatible, allow_nan=False)) == incompatible


def test_family_counts_recorded_terminal_failures(family_runs, tmp_path, monkeypatch):
    project, _, _, _ = family_runs
    original = RuleBasedManifest.run_model

    def fail_last_trial(manifest, data, round_params):
        if round_params['entry_return'] == 0.002:
            raise StrictModeError('Recorded family trial failure')
        return original(manifest, data, round_params)

    monkeypatch.setattr(RuleBasedManifest, 'run_model', fail_last_trial)
    config = _config()
    config['sfd']['params']['entry_return'] = [0.0, 0.001, 0.002]
    directory = tmp_path / 'failure'
    _experiment(config, _bars(), directory, project)
    report = family_acceptance_report([directory], project_root=project)
    assert report['coverage']['known_failed_trials'] == 1
    assert report['coverage']['recorded_trials'] == 2
    assert len(report['runs'][0]['terminal_trials']) == 3
    (directory / 'results.csv').unlink()
    assert family_acceptance_report([directory], project_root=project)['coverage']['runs_without_terminal_inventory'] == 1


@pytest.mark.parametrize('damage', ('gzip', 'json', 'blob', 'returns', 'csv', 'manifest'))
def test_family_propagates_integrity_failures_without_writes(family_runs, tmp_path, damage):
    project, paths, _, _ = family_runs
    directory = _copy_run(paths[0], tmp_path / damage)
    if damage == 'gzip':
        path = storage.round_path(directory)
        path.write_bytes(path.read_bytes()[:-1])
    elif damage == 'json':
        storage.round_path(directory).write_bytes(gzip.compress(b'not json\n'))
    elif damage == 'blob':
        records = _records(directory)
        records[0]['preds'] = records[0]['folds'][0]['preds']
        _write_records(directory, records)
        path = next((directory / 'round_signals').glob('*.json.gz'))
        path.write_bytes(path.read_bytes()[:-1])
    elif damage == 'returns':
        records = _records(directory)
        values = records[0]['folds'][0]['net_returns']
        values[0] = next(value for value in values if value != values[0])
        _write_records(directory, records)
    elif damage == 'csv':
        with (directory / 'results.csv').open(newline='') as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        with (directory / 'results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows[:-1])
    else:
        metadata = json.loads((directory / 'metadata.json').read_text())
        metadata['yaml_reference']['sfd']['manifest']['backtest']['fee_bps'] = 3.0
        (directory / 'metadata.json').write_text(json.dumps(metadata))
    before = _files(directory, project / STORE_RELATIVE)
    with pytest.raises((ValueError, FileNotFoundError)):
        family_acceptance_report([paths[0], directory], project_root=project)
    assert _files(directory, project / STORE_RELATIVE) == before


def test_family_verifies_lineage_without_the_index(family_runs, tmp_path):
    project, paths, root_id, child_id = family_runs
    store = tmp_path / STORE_RELATIVE
    shutil.copytree(project / STORE_RELATIVE, store)
    (store / 'index.json').write_text('not a valid cache')
    before = _files(paths[1], store)
    report = family_acceptance_report([paths[1]], project_root=tmp_path)
    assert [node['manifest_id'] for node in report['runs'][0]['lineage']] == [root_id, child_id]
    assert _files(paths[1], store) == before
    parent = store / (root_id.removeprefix('sha256:') + '.yaml')
    document = YAML().load(parent.read_text())
    document['lineage']['parent_id'] = child_id
    with parent.open('w') as stream:
        YAML().dump(document, stream)
    with pytest.raises(ValueError, match='cyclic'):
        family_acceptance_report([paths[1]], project_root=tmp_path)
    document['sfd']['manifest']['backtest']['fee_bps'] = 3.0
    with parent.open('w') as stream:
        YAML().dump(document, stream)
    with pytest.raises(ValueError, match='identity mismatch'):
        family_acceptance_report([paths[1]], project_root=tmp_path)
