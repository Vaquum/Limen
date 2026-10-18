import copy
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from ruamel.yaml import YAML

from limen.experiment import _round_storage as storage
from limen.experiment.family_acceptance_report import family_acceptance_report
from limen.metrics import deflated_sharpe_ratio, probability_of_backtest_overfitting
from limen.yaml.store import STORE_RELATIVE, commit_manifest
from tests.test_walk_forward_uel import _bars, _config, _loop, _run


def _experiment(config, bars, directory, project, *, parent=None):
    manifest = project / 'experiment.yaml'
    with manifest.open('w') as stream:
        YAML().dump(config, stream)
    identity, _ = commit_manifest(manifest, project, parent_id=parent)
    loop = _loop(config, bars, directory)
    _run(loop, n_permutations=len(config['sfd']['params']['entry_return']))
    return identity


def _tracks(directory):
    frame = pl.read_parquet(directory / 'trial_returns.parquet')
    return {trial: frame.filter(pl.col('trial') == trial).sort('bar')['net_return'].to_numpy()
            for trial in frame['trial'].unique(maintain_order=True).to_list()}


def _copy_run(source, target):
    shutil.copytree(source, target)
    return target


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
        'func': 'limen.indicators.window_return', 'params': {'period': 2},
    })
    varied_config['sfd']['params']['entry_return'] = [0.0]
    _experiment(varied_config, _bars(), varied, project, parent=root_id)
    return project, [parent, child, repeated, varied], root_id, child_id


def test_family_combines_comparable_runs(family_runs):
    project, paths, root_id, child_id = family_runs
    document = (Path(__file__).parents[1] / 'docs/Benchmark.md').read_text()
    example = document.split('### Comparing a recorded experiment family', 1)[1].split('```python', 1)[1].split('```', 1)[0]
    namespace = {'run_directories': list(reversed(paths)), 'project_root': project}
    exec(example, namespace)
    report = namespace['report']
    assert report == family_acceptance_report(paths, project_root=project)
    assert len(report['comparisons']) == 1
    group = report['comparisons'][0]
    parent_tracks, child_tracks, varied_tracks = [_tracks(paths[index]) for index in (0, 1, 3)]
    tracks = [*parent_tracks.values(), list(child_tracks.values())[1], *varied_tracks.values()]
    matrix = np.stack(tracks)
    sharpes = matrix.mean(axis=1) / matrix.std(axis=1, ddof=1)
    variance = float(sharpes.var(ddof=1))
    winner = int(np.argmax(sharpes))
    assert group['n_trials'] == 4
    assert group['n_bars'] == matrix.shape[1]
    assert group['winner_sharpe'] == pytest.approx(float(sharpes[winner]))
    assert group['trial_sharpe_variance'] == pytest.approx(variance)
    assert group['deflated_sharpe_probability'] == pytest.approx(
        deflated_sharpe_ratio(matrix[winner], n_trials=4, trial_sharpe_variance=variance),
    )
    assert group['pbo'] == probability_of_backtest_overfitting(matrix, n_blocks=2)
    assert group['errors'] == {}
    assert group['members'][0] == {'directory': str(paths[0]), 'trial': next(iter(parent_tracks))}
    child_run = next(run for run in report['runs'] if run['directory'] == str(paths[1]))
    assert [item['manifest_id'] for item in child_run['lineage']] == [root_id, child_id]
    assert all(item['available'] for item in child_run['lineage'])
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert not (project / 'acceptance_report.json').exists()


def test_family_preserves_variations_and_identifies_repeats(family_runs):
    project, paths, _, _ = family_runs
    report = family_acceptance_report([*paths, paths[0]], project_root=project)
    assert report['coverage'] == {'recorded_trials': 7, 'distinct_compared_trials': 4, 'repeats': 3, 'unavailable': 0}
    assert report['duplicate_inputs'] == [str(paths[0])]
    repeats = [trial for trial in report['trials'] if trial['repeat_of'] is not None]
    assert len(repeats) == 3
    assert all(trial['repeat_reason'] and trial['comparison'] for trial in repeats)
    assert all(trial['repeat_of']['directory'] != str(paths[2]) for trial in repeats)
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


def test_family_separates_signed_execution(family_runs, tmp_path):
    project, paths, _, _ = family_runs
    config = _config()
    config['metadata']['name'] = 'family_signed'
    config['sfd']['manifest']['backtest']['product'] = {
        'kind': 'cash_spot', 'instrument': 'BTCUSDT', 'base_currency': 'BTC',
        'quote_currency': 'USDT', 'quantity_step': 1e-9, 'min_notional': 0.0,
    }
    signed = tmp_path / 'signed'
    _experiment(config, _bars(), signed, project)
    report = family_acceptance_report([paths[0], signed], project_root=project)
    assert len(report['comparisons']) == 2
    assert {group['evidence']['folds'][0]['return_semantics'] for group in report['comparisons']} == {'snapshot', 'signed_execution'}
    assert all(group['n_trials'] == 2 for group in report['comparisons'])
    signed_group = next(group for group in report['comparisons'] if group['evidence']['folds'][0]['return_semantics'] == 'signed_execution')
    assert signed_group['evidence']['folds'][0]['product']['quote_currency'] == 'USDT'
    assert signed_group['pbo'] == probability_of_backtest_overfitting(np.stack(list(_tracks(signed).values())), n_blocks=2)


def test_family_rejects_corruption_and_explains_metric_limits(family_runs, tmp_path):
    project, paths, _, _ = family_runs
    damaged = _copy_run(paths[0], tmp_path / 'duplicate_bar')
    frame = pl.read_parquet(damaged / 'trial_returns.parquet')
    pl.concat([frame, frame.head(1)]).write_parquet(damaged / 'trial_returns.parquet')
    with pytest.raises(ValueError, match='unique contiguous'):
        family_acceptance_report([damaged], project_root=project)
    missing = tmp_path / 'missing_returns'
    missing.mkdir()
    report = family_acceptance_report([missing], project_root=project)
    assert report['comparisons'] == []
    assert any('coverage is unknown' in error for error in report['runs'][0]['errors'])
    blocked = family_acceptance_report([paths[0]], project_root=project, n_blocks=4)['comparisons'][0]
    assert blocked['pbo'] is None
    assert 'equal contiguous blocks' in blocked['errors']['pbo']
    assert isinstance(blocked['deflated_sharpe_probability'], float)
    for count in (1, 3, True, 2.0):
        with pytest.raises(ValueError, match='even integer'):
            family_acceptance_report(paths, project_root=project, n_blocks=count)
    with pytest.raises(ValueError, match='at least one'):
        family_acceptance_report([], project_root=project)
