import gzip
from limen.experiment import _round_storage as _storage
import copy
import json
import math
from datetime import date, timezone
from itertools import combinations, pairwise
from pathlib import Path
from statistics import NormalDist
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from limen.metrics import deflated_sharpe_ratio, probability_of_backtest_overfitting
from limen.data.utils import split_walk_forward
from limen.experiment import Manifest
from limen.experiment.acceptance_report import acceptance_report
from limen.yaml import CompiledSFD, parse, validate
from tests.test_walk_forward_uel import _GEOMETRY, _bars, _config, _loop, _run


_FIXTURE = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'


def _market_returns():
    close = pl.read_parquet(_FIXTURE)['close'].to_numpy()
    return np.diff(close) / close[:-1]


def _reference_dsr(track, n_trials, variance):
    normal = NormalDist()
    sharpe = float(track.mean() / track.std(ddof=1))
    centered = track - track.mean()
    scale = float(np.sqrt(np.mean(centered ** 2)))
    skew = float(np.mean((centered / scale) ** 3))
    kurtosis = float(np.mean((centered / scale) ** 4))
    benchmark = 0.0
    if n_trials > 1 and variance > 0.0:
        euler_gamma = 0.5772156649015329
        benchmark = math.sqrt(variance) * (
            (1 - euler_gamma) * normal.inv_cdf(1 - 1 / n_trials)
            + euler_gamma * normal.inv_cdf(1 - 1 / (n_trials * math.e))
        )
    corrected_variance = 1 - skew * sharpe + (kurtosis - 1) * sharpe ** 2 / 4
    return normal.cdf((sharpe - benchmark) * math.sqrt((len(track) - 1) / corrected_variance))


def _reference_pbo(matrix, n_blocks):
    blocks = np.array_split(np.arange(matrix.shape[1]), n_blocks)
    losses = []
    for selected in combinations(range(n_blocks), n_blocks // 2):
        in_sample = np.concatenate([blocks[index] for index in selected])
        out_sample = np.concatenate([blocks[index] for index in range(n_blocks) if index not in selected])
        fit = matrix[:, in_sample]
        held_out = matrix[:, out_sample]
        winner = int(np.argmax(fit.mean(axis=1) / fit.std(axis=1, ddof=1)))
        scores = held_out.mean(axis=1) / held_out.std(axis=1, ddof=1)
        lower = int(np.count_nonzero(scores < scores[winner]))
        ties = int(np.count_nonzero(scores == scores[winner]))
        rank = lower + (ties + 1) / 2
        losses.append(rank / (matrix.shape[0] + 1) <= 0.5)
    return sum(losses) / len(losses)


def test_deflated_sharpe_laws():
    # Every observation comes from the recorded hourly fixture; no noise is generated.
    matrix = _market_returns()[:768].reshape(4, 192)
    track = matrix[0]
    sharpes = matrix.mean(axis=1) / matrix.std(axis=1, ddof=1)
    variance = float(sharpes.var(ddof=1))
    probabilities = [deflated_sharpe_ratio(track, n_trials=count, trial_sharpe_variance=variance)
                     for count in (1, 2, 4, 16)]
    assert all(0.0 <= value <= 1.0 for value in probabilities)
    assert all(before >= after for before, after in pairwise(probabilities))
    for count, observed in zip((1, 2, 4, 16), probabilities, strict=True):
        assert observed == pytest.approx(_reference_dsr(track, count, variance), abs=1e-14)
    assert deflated_sharpe_ratio(track, n_trials=4, trial_sharpe_variance=0.0) == pytest.approx(
        _reference_dsr(track, 1, 0.0), abs=1e-14,
    )
    for invalid in (track[:0], track[:3], track[:1].repeat(4), matrix):
        with pytest.raises(ValueError):
            deflated_sharpe_ratio(invalid, n_trials=4, trial_sharpe_variance=variance)
    for count in (0, -1, True, 2.0):
        with pytest.raises(ValueError):
            deflated_sharpe_ratio(track, n_trials=count, trial_sharpe_variance=variance)
    for invalid_variance in (-1.0, float('inf'), float('nan')):
        with pytest.raises(ValueError):
            deflated_sharpe_ratio(track, n_trials=4, trial_sharpe_variance=invalid_variance)


def test_pbo_laws():
    returns = _market_returns()
    matrix = returns[:768].reshape(4, 192)
    observed = probability_of_backtest_overfitting(matrix, n_blocks=8)
    assert 0.0 <= observed <= 1.0
    assert observed == _reference_pbo(matrix, 8)
    # Subsample actual observations to expose the dominance and exact-tie contracts.
    dominating = np.stack([returns[returns > 0][:64], returns[returns < 0][:64]])
    assert probability_of_backtest_overfitting(dominating, n_blocks=8) == 0.0
    tied = np.stack([matrix[0], matrix[0]])
    assert probability_of_backtest_overfitting(tied, n_blocks=8) == 1.0
    reversed_trials = matrix[::-1].copy()
    assert probability_of_backtest_overfitting(reversed_trials, n_blocks=8) == observed
    for invalid in (matrix[0], matrix[:1], matrix[:, :7], matrix[:, :15], matrix[:, :0]):
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(invalid, n_blocks=8)
    constant_trial = np.stack([matrix[0, :64], matrix[0, :1].repeat(64)])
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(constant_trial, n_blocks=8)
    for count in (0, 1, 3, True, 8.0):
        with pytest.raises(ValueError):
            probability_of_backtest_overfitting(matrix, n_blocks=count)


@pytest.fixture(scope='module')
def acceptance_runs(tmp_path_factory):
    outputs = []
    document = (Path(__file__).parents[1] / 'docs/Experiment-Manifest.md').read_text()
    example = document.split('### Recorded acceptance example', 1)[1].split('```yaml', 1)[1].split('```', 1)[0]
    canonical, errors = parse(example)
    assert not errors
    assert canonical['metadata']['name'] == 'recorded_acceptance'
    assert canonical['sfd']['manifest']['data_source']['params']['file_path_or_url'] == 'inputs/' + _FIXTURE.name
    for declared in ('none', 'canonical', 'strict'):
        config = copy.deepcopy(canonical)
        if declared == 'none':
            config['sfd']['manifest'].pop('acceptance')
        elif declared == 'strict':
            config['sfd']['manifest']['acceptance'] = {
                'min_deflated_sharpe_probability': 1.0,
                'max_pbo': 0.0,
            }
        path = tmp_path_factory.mktemp(f'acceptance_{declared}')
        loop = _loop(config, pl.read_parquet(_FIXTURE), path)
        _run(loop)
        outputs.append((loop, config, path))
    return outputs


def test_acceptance_block_and_verdicts(acceptance_runs, tmp_path):
    config = copy.deepcopy(acceptance_runs[2][1])
    assert validate(config).valid
    native = Manifest().set_split_walk_forward(**_GEOMETRY).set_acceptance(
        min_deflated_sharpe_probability=1.0, max_pbo=0.0,
    )
    assert native.acceptance == CompiledSFD(config).manifest().acceptance
    for key in ('min_deflated_sharpe_probability', 'max_pbo'):
        partial = copy.deepcopy(config)
        partial['sfd']['manifest']['acceptance'] = {key: 0.5}
        assert validate(partial).valid
        for value in (-0.01, 1.01, float('nan'), float('inf'), True, '0.5', '{threshold}'):
            changed = copy.deepcopy(config)
            changed['sfd']['manifest']['acceptance'][key] = value
            assert not validate(changed).valid
    for declaration in (None, {}, 0.5, {**config['sfd']['manifest']['acceptance'], 'extra': 0.5}):
        changed = copy.deepcopy(config)
        changed['sfd']['manifest']['acceptance'] = declaration
        assert not validate(changed).valid
    changed = copy.deepcopy(config)
    changed['sfd']['manifest'].pop('split_walk_forward')
    assert not validate(changed).valid
    with pytest.raises(ValueError, match='split_walk_forward'):
        Manifest().set_acceptance(max_pbo=0.5)
    for declaration in ({}, {'max_pbo': True}, {'max_pbo': 1.1}):
        with pytest.raises(ValueError, match='acceptance'):
            Manifest().set_split_walk_forward(**_GEOMETRY).set_acceptance(**declaration)
    loop, _, path = acceptance_runs[2]
    report = json.loads((path / 'acceptance_report.json').read_text())
    assert report['thresholds'] == native.acceptance
    assert report['verdicts'] == {
        'min_deflated_sharpe_probability': report['deflated_sharpe_probability'] >= 1.0,
        'max_pbo': report['pbo'] <= 0.0,
    }
    assert not report['verdicts']['min_deflated_sharpe_probability']
    assert loop.experiment_log.height == 2
    (tmp_path / 'trial_returns.parquet').write_bytes((path / 'trial_returns.parquet').read_bytes())
    (tmp_path / 'round_data.jsonl').write_text(_storage.text(_storage.round_path(path)))
    permissive = acceptance_report(tmp_path, acceptance={
        'min_deflated_sharpe_probability': 0.0, 'max_pbo': 1.0,
    })
    assert permissive['verdicts'] == {
        'min_deflated_sharpe_probability': True, 'max_pbo': True,
    }


@pytest.mark.parametrize('key', ('min_deflated_sharpe_probability', 'max_pbo'))
def test_acceptance_rejects_oversized_integer(key):
    config = _config()
    config['sfd']['manifest']['acceptance'] = {key: 10 ** 400}
    result = validate(config)
    assert not result.valid
    assert any('acceptance' in error.path for error in result.errors)
    with pytest.raises(ValueError, match='acceptance'):
        Manifest().set_split_walk_forward(**_GEOMETRY).set_acceptance(**{key: 10 ** 400})


def test_pbo_requires_recorded_synchronous_dates(tmp_path):
    bars = _bars()
    loop = _loop(_config(), bars, tmp_path)
    _run(loop)
    artifact = pl.read_parquet(tmp_path / 'trial_returns.parquet')
    trials = artifact['trial'].unique(maintain_order=True).to_list()
    matrix = np.stack([artifact.filter(pl.col('trial') == trial)['net_return'].to_numpy()
                       for trial in trials])
    path = _storage.round_path(tmp_path)
    records = [json.loads(line) for line in _storage.text(path).splitlines()]
    for record in records:
        for fold in record['folds']:
            fold['alignment'].pop('test_datetimes', None)
    path.write_bytes(b''.join(gzip.compress((json.dumps(record) + '\n').encode(), mtime=0) for record in records))
    absent = acceptance_report(tmp_path, acceptance={'max_pbo': 0.5})
    assert absent['pbo'] is None
    assert absent['verdicts']['max_pbo'] is None
    assert absent['errors']['pbo']
    assert isinstance(absent['deflated_sharpe_probability'], float)
    source_dates = [value.isoformat() for value in bars['datetime']]
    for record in records:
        for fold in record['folds']:
            alignment = fold['alignment']
            dates = [value for value in source_dates
                     if alignment['first_test_datetime'] <= value <= alignment['last_test_datetime']
                     and value not in alignment['missing_datetimes']]
            assert len(dates) == len(fold['net_returns'])
            alignment['test_datetimes'] = dates
    path.write_bytes(b''.join(gzip.compress((json.dumps(record) + '\n').encode(), mtime=0) for record in records))
    aligned = acceptance_report(tmp_path, acceptance={'max_pbo': 0.5})
    assert aligned['pbo'] == _reference_pbo(matrix, 2)
    assert aligned['deflated_sharpe_probability'] == absent['deflated_sharpe_probability']
    assert aligned['errors'] == {}
    valid_records = copy.deepcopy(records)
    for fold in records[1]['folds']:
        dates = fold['alignment']['test_datetimes']
        start = source_dates.index(dates[0]) - 1
        adjacent = source_dates[start:start + len(dates)]
        assert len(adjacent) == len(dates) and adjacent != dates
        fold['alignment']['test_datetimes'] = adjacent
    path.write_bytes(b''.join(gzip.compress((json.dumps(record) + '\n').encode(), mtime=0) for record in records))
    shifted = acceptance_report(tmp_path, acceptance={'max_pbo': 0.5})
    assert shifted['pbo'] is None
    assert shifted['verdicts']['max_pbo'] is None
    assert shifted['errors']['pbo']
    assert shifted['deflated_sharpe_probability'] == aligned['deflated_sharpe_probability']
    records = valid_records
    recorded_returns = records[1]['folds'][0]['net_returns']
    recorded_returns[1] = {'recorded_return': recorded_returns[1]}
    path.write_bytes(b''.join(gzip.compress((json.dumps(record) + '\n').encode(), mtime=0) for record in records))
    corrupt = acceptance_report(tmp_path, acceptance={'max_pbo': 0.5})
    assert corrupt['pbo'] is None
    assert corrupt['verdicts']['max_pbo'] is None
    assert 'numeric' in corrupt['errors']['pbo']
    assert corrupt['deflated_sharpe_probability'] == aligned['deflated_sharpe_probability']
    assert json.loads((tmp_path / 'acceptance_report.json').read_text()) == corrupt
    assert 'numeric' in (tmp_path / 'acceptance_report.md').read_text()


def test_pbo_timestamp_identity_survives_recorded_dst_transition(tmp_path):
    bars = pl.read_parquet(_FIXTURE).filter(pl.col('datetime').dt.date() <= date(2024, 10, 28)).tail(384)
    config = _config()
    config['sfd']['manifest']['split_walk_forward']['test_bars'] = 49
    loop = _loop(config, bars, tmp_path)
    _run(loop)
    artifact = pl.read_parquet(tmp_path / 'trial_returns.parquet')
    trials = artifact['trial'].unique(maintain_order=True).to_list()
    matrix = np.stack([artifact.filter(pl.col('trial') == trial)['net_return'].to_numpy()
                       for trial in trials])
    path = _storage.round_path(tmp_path)
    records = [json.loads(line) for line in _storage.text(path).splitlines()]
    helsinki = ZoneInfo('Europe/Helsinki')
    lexical_disorder = False
    for trial_index, record in enumerate(records):
        for fold in record['folds']:
            alignment = fold['alignment']
            dates = [value for value in bars['datetime']
                     if alignment['first_test_datetime'] <= value.isoformat() <= alignment['last_test_datetime']
                     and value.isoformat() not in alignment['missing_datetimes']]
            assert len(dates) == len(fold['net_returns'])
            instants = [value.replace(tzinfo=timezone.utc) if value.tzinfo is None
                        else value.astimezone(timezone.utc) for value in dates]
            localized = [value.astimezone(helsinki) for value in instants]
            if any(before.utcoffset() != after.utcoffset() for before, after in pairwise(localized)):
                lexical_disorder = True
                assert [value.isoformat() for value in localized] != sorted(value.isoformat() for value in localized)
            zone = helsinki if trial_index == 0 else timezone.utc
            alignment['test_datetimes'] = [value.astimezone(zone).isoformat() for value in instants]
    assert lexical_disorder
    path.write_bytes(b''.join(gzip.compress((json.dumps(record) + '\n').encode(), mtime=0) for record in records))
    aligned = acceptance_report(tmp_path, acceptance={'max_pbo': 0.5})
    assert aligned['pbo'] == _reference_pbo(matrix, 2)
    assert aligned['errors'] == {}
    original = records[1]['folds'][0]['alignment']['test_datetimes']
    for invalid in (original[::-1], [original[0], *original[:-1]]):
        records[1]['folds'][0]['alignment']['test_datetimes'] = invalid
        path.write_bytes(b''.join(gzip.compress((json.dumps(record) + '\n').encode(), mtime=0) for record in records))
        rejected = acceptance_report(tmp_path, acceptance={'max_pbo': 0.5})
        assert rejected['pbo'] is None
        assert rejected['verdicts']['max_pbo'] is None
        assert rejected['errors']['pbo']
        assert rejected['deflated_sharpe_probability'] == aligned['deflated_sharpe_probability']


def test_report_end_to_end(acceptance_runs):
    expected_keys = {
        'n_trials', 'n_bars', 'n_blocks', 'winner_trial', 'winner_sharpe',
        'trial_sharpe_variance', 'deflated_sharpe_probability', 'pbo',
        'thresholds', 'verdicts', 'errors', 'pbo_combinations', 'pbo_block_bars', 'pbo_resolution',
    }
    for loop, config, path in acceptance_runs:
        report = json.loads((path / 'acceptance_report.json').read_text())
        assert set(report) == expected_keys
        artifact = pl.read_parquet(path / 'trial_returns.parquet')
        trials = artifact['trial'].unique(maintain_order=True).to_list()
        folds = split_walk_forward(pl.read_parquet(_FIXTURE), **config['sfd']['manifest']['split_walk_forward'])
        expected_dates = [[value.isoformat() for value in test['datetime'].slice(1)] for _, test in folds]
        records = [json.loads(line) for line in _storage.text(_storage.round_path(path)).splitlines()]
        assert [record['round_id'] for record in records] == trials
        for record in records:
            for nested, dates in zip(record['folds'], expected_dates, strict=True):
                assert nested['alignment']['test_datetimes'] == dates
                assert len(nested['preds']) == len(nested['net_returns']) == len(dates)
        matrix = np.stack([artifact.filter(pl.col('trial') == trial)['net_return'].to_numpy()
                           for trial in trials])
        sharpes = matrix.mean(axis=1) / matrix.std(axis=1, ddof=1)
        winner = int(np.argmax(sharpes))
        variance = float(sharpes.var(ddof=1))
        assert report['n_trials'] == len(trials) == loop.experiment_log.height == 2
        assert report['n_bars'] == matrix.shape[1] == 96
        assert report['n_blocks'] == 2
        assert report['winner_trial'] == trials[winner]
        assert report['winner_sharpe'] == pytest.approx(sharpes[winner])
        assert report['trial_sharpe_variance'] == pytest.approx(variance)
        assert report['deflated_sharpe_probability'] == pytest.approx(
            _reference_dsr(matrix[winner], len(trials), variance), abs=1e-14,
        )
        assert report['pbo'] == _reference_pbo(matrix, 2)
        assert report['errors'] == {}
        human = (path / 'acceptance_report.md').read_text()
        assert 'Deflated Sharpe' in human
        assert 'PBO' in human
        assert trials[winner] in human
        if 'acceptance' not in config['sfd']['manifest']:
            assert report['thresholds'] == report['verdicts'] == {}
    benchmark = (Path(__file__).parents[1] / 'docs/Benchmark.md').read_text()
    assert 'walk-forward' in benchmark
    assert 'does not claim' not in benchmark


def test_report_artifact_gating_and_unavailable_statistics(tmp_path):
    assert acceptance_report(tmp_path) is None
    assert not (tmp_path / 'acceptance_report.json').exists()
    config = _config()
    config['sfd']['params']['entry_return'] = [0.0]
    config['sfd']['manifest']['acceptance'] = {
        'min_deflated_sharpe_probability': 0.5, 'max_pbo': 0.5,
    }
    loop = _loop(config, _bars(), tmp_path)
    _run(loop, n_permutations=1)
    report = json.loads((tmp_path / 'acceptance_report.json').read_text())
    assert loop.experiment_log.height == report['n_trials'] == 1
    assert report['pbo'] is None
    assert report['verdicts']['max_pbo'] is None
    assert report['errors']['pbo']
    assert (tmp_path / 'acceptance_report.md').is_file()


def test_all_flat_report_is_unavailable_without_aborting(tmp_path):
    config = _config()
    config['sfd']['params']['entry_return'] = [1.0]
    config['sfd']['manifest']['acceptance'] = {
        'min_deflated_sharpe_probability': 0.5, 'max_pbo': 0.5,
    }
    loop = _loop(config, _bars(), tmp_path)
    _run(loop, n_permutations=1)
    report = json.loads((tmp_path / 'acceptance_report.json').read_text())
    returns = pl.read_parquet(tmp_path / 'trial_returns.parquet')
    assert returns['net_return'].eq(0.0).all()
    assert loop.experiment_log.height == 1
    assert report['deflated_sharpe_probability'] is report['pbo'] is None
    assert report['verdicts'] == {
        'min_deflated_sharpe_probability': None, 'max_pbo': None,
    }
    assert set(report['errors']) == {'deflated_sharpe_probability', 'pbo'}


def test_default_run_writes_no_acceptance_report(tmp_path):
    config = _config()
    config['sfd']['manifest'].pop('split_walk_forward')
    config['sfd']['manifest']['split_dates'] = {
        'train_start': '2024-01-01', 'train_end': '2024-01-08',
        'val_start': '2024-01-08', 'val_end': '2024-01-11',
        'test_start': '2024-01-11', 'test_end': '2024-01-16',
    }
    loop = _loop(config, _bars(), tmp_path)
    _run(loop, n_permutations=1)
    assert loop.experiment_log.height == 1
    assert not (tmp_path / 'acceptance_report.json').exists()
    assert not (tmp_path / 'acceptance_report.md').exists()


def _artifact_snapshot(path):
    return {str(file.relative_to(path)): (file.read_bytes(), file.stat().st_mtime_ns)
            for file in path.rglob('*') if file.is_file()}


@pytest.fixture(scope='module')
def pbo_runs(acceptance_runs, tmp_path_factory):
    from types import SimpleNamespace
    from limen.experiment import UniversalExperimentLoop
    from limen.yaml import build_search_strategy
    from limen.yaml.compiler import build_manifest

    outputs = {}
    for kind in ('native', 'compiled_only', 'sparse', 'indivisible'):
        config = copy.deepcopy(acceptance_runs[0][1]) if kind != 'indivisible' else _config()
        config['sfd']['manifest']['acceptance'] = {'pbo_blocks': 8}
        if kind == 'sparse':
            config['sfd']['params']['entry_return'] = [0.0, 0.004]
            config['sfd']['manifest']['acceptance']['max_pbo'] = 0.2
        path = tmp_path_factory.mktemp(f'pbo_{kind}')
        if kind == 'native':
            manifest = build_manifest(config)
            sfd = SimpleNamespace(__name__=__name__, params=lambda declaration=config: declaration['sfd']['params'], manifest=lambda frozen=manifest: frozen)
        else:
            sfd = CompiledSFD(config)
        loop = UniversalExperimentLoop(sfd=sfd, data=pl.read_parquet(_FIXTURE) if kind != 'indivisible' else _bars(),
                                       experiment_dir=path, search_strategy=build_search_strategy(config), checkpoint_interval=1)
        _run(loop)
        outputs[kind] = (loop, config, path)
    return outputs


def test_pbo_configuration_routes(acceptance_runs, pbo_runs, tmp_path):
    from click.testing import CliRunner
    from limen.cli.main import cli
    from limen.experiment import UniversalExperimentLoop
    from limen.experiment.acceptance_report import read_acceptance
    from limen.yaml import build_search_strategy

    base = copy.deepcopy(acceptance_runs[0][1])
    for count in (2, 4, 6, 8, 10, 12, 10**1000):
        config = copy.deepcopy(base)
        config['sfd']['manifest']['acceptance'] = {'pbo_blocks': count}
        assert validate(config).valid
        yaml_path = tmp_path / 'valid.yaml'
        yaml_path.write_text(json.dumps(config))
        assert CliRunner().invoke(cli, ['validate', str(yaml_path)]).exit_code == 0
        assert Manifest().set_split_walk_forward(**_GEOMETRY).set_acceptance(pbo_blocks=count).acceptance == {'pbo_blocks': count}
        assert type(read_acceptance({'pbo_blocks': count})['pbo_blocks']) is int
        for policy in ({}, {'alignment': True}, {'fold_details': True}):
            config['uel']['recording'] = policy
            assert not validate(config).valid
    for value in (True, False, 0, -2, 1, 3, 9, 8.0, '8', '{blocks}'):
        config = copy.deepcopy(base)
        config['sfd']['manifest']['acceptance'] = {'pbo_blocks': value}
        validation = validate(config)
        assert not validation.valid
        assert any(error.path == 'sfd.manifest.acceptance' for error in validation.errors)
        yaml_path = tmp_path / 'invalid.yaml'
        yaml_path.write_text(json.dumps(config))
        assert CliRunner().invoke(cli, ['validate', str(yaml_path)]).exit_code != 0
        with pytest.raises(ValueError, match='pbo_blocks'):
            Manifest().set_split_walk_forward(**_GEOMETRY).set_acceptance(pbo_blocks=value)
        snapshot = _artifact_snapshot(pbo_runs['compiled_only'][2])
        with pytest.raises(ValueError, match='pbo_blocks'):
            acceptance_report(pbo_runs['compiled_only'][2], acceptance={'pbo_blocks': value})
        assert _artifact_snapshot(pbo_runs['compiled_only'][2]) == snapshot
    config = copy.deepcopy(pbo_runs['compiled_only'][1])
    for has_search, has_directory in ((False, True), (True, False)):
        path = tmp_path / f'missing_{has_search}_{has_directory}'
        loop = UniversalExperimentLoop(sfd=CompiledSFD(config), data=_bars(),
                                       search_strategy=build_search_strategy(config) if has_search else None,
                                       experiment_dir=path if has_directory else None)
        before = _artifact_snapshot(tmp_path)
        with pytest.raises(ValueError, match='search_strategy and experiment_dir'):
            loop.run('admission', progress_bar=False)
        assert _artifact_snapshot(tmp_path) == before
    reference = copy.deepcopy(config)
    reference['sfd']['manifest']['acceptance']['pbo_blocks'] = 2
    path = tmp_path / 'conflict'
    loop = UniversalExperimentLoop(sfd=CompiledSFD(config), data=_bars(), experiment_dir=path,
                                   search_strategy=build_search_strategy(config), yaml_reference=reference)
    with pytest.raises(ValueError, match='Conflicting'):
        _run(loop)
    assert not path.exists()
    loop._yaml_reference = config
    loop.manifest.acceptance = {'pbo_blocks': 9}
    with pytest.raises(ValueError, match='pbo_blocks'):
        _run(loop)
    assert not path.exists()
    for kind in ('native', 'compiled_only'):
        native_loop, _, directory = pbo_runs[kind]
        assert json.loads((directory / 'metadata.json').read_text())['pbo_blocks'] == 8
        assert json.loads((directory / 'acceptance_report.json').read_text())['thresholds'] == {}
        assert native_loop._compiled_yaml is None if kind == 'native' else native_loop._compiled_yaml is not None
    minimal = _loop(config, pl.read_parquet(_FIXTURE), tmp_path / 'minimal')
    minimal.run('minimal', n_permutations=2, prep_each_round=True, progress_bar=False,
                recording={'alignment': True, 'fold_details': True})
    assert json.loads((tmp_path / 'minimal/acceptance_report.json').read_text())['pbo'] == 6 / 7
    before = _artifact_snapshot(tmp_path / 'minimal')
    with pytest.raises(ValueError, match='lighter recording'):
        minimal.run('minimal', resume=True, progress_bar=False)
    assert _artifact_snapshot(tmp_path / 'minimal') == before
    omitted = copy.deepcopy(base)
    omitted['sfd']['manifest']['acceptance'] = {'max_pbo': 0.2}
    omitted['uel'].pop('recording', None)
    standard = _loop(omitted, pl.read_parquet(_FIXTURE), tmp_path / 'standard', search=False)
    _run(standard, n_permutations=1)
    assert not (tmp_path / 'standard/metadata.json').exists()
    assert json.loads((tmp_path / 'standard/acceptance_report.json').read_text())['pbo'] is None


def test_pbo_configuration_reports(acceptance_runs, pbo_runs, tmp_path):
    default_path = acceptance_runs[0][2]
    default = json.loads((default_path / 'acceptance_report.json').read_text())
    path = pbo_runs['compiled_only'][2]
    report = json.loads((path / 'acceptance_report.json').read_text())
    with pytest.raises(ValueError, match='acceptance'):
        acceptance_report(path, acceptance={})
    frame = pl.read_parquet(path / 'trial_returns.parquet')
    matrix = np.stack([group['net_return'].to_numpy() for group in frame.partition_by('trial', maintain_order=True)])
    assert report['pbo'] == _reference_pbo(matrix, 8) == probability_of_backtest_overfitting(matrix, n_blocks=8) == 6 / 7
    assert (report['n_bars'], report['n_blocks'], report['pbo_block_bars'], report['pbo_combinations'], report['pbo_resolution']) == (96, 8, 12, 70, 1 / 70)
    assert frame.equals(pl.read_parquet(default_path / 'trial_returns.parquet'))
    for field in ('winner_trial', 'winner_sharpe', 'trial_sharpe_variance', 'deflated_sharpe_probability'):
        assert report[field] == default[field]
    markdown = (path / 'acceptance_report.md').read_text()
    assert 'CSCV blocks: 8' in markdown and 'Nominal combinations: 70' in markdown
    assert 'Nominal counting step (float approximation): 0.014285714285714285' in markdown and 'Null is not pass' in markdown
    sparse_path = pbo_runs['sparse'][2]
    sparse = json.loads((sparse_path / 'acceptance_report.json').read_text())
    assert sparse['pbo'] is sparse['verdicts']['max_pbo'] is None
    assert sparse['deflated_sharpe_probability'] is not None and sparse['pbo_block_bars'] == 12
    assert 'half sample' in sparse['errors']['pbo']
    sparse_frame = pl.read_parquet(sparse_path / 'trial_returns.parquet')
    sparse_matrix = np.stack([group['net_return'].to_numpy() for group in sparse_frame.partition_by('trial', maintain_order=True)])
    assert probability_of_backtest_overfitting(sparse_matrix, n_blocks=2) == 0.0
    bad = json.loads((pbo_runs['indivisible'][2] / 'acceptance_report.json').read_text())
    assert bad['n_bars'] == 94 and bad['pbo_block_bars'] is bad['pbo'] is None
    assert 'n_bars=94, n_blocks=8; supported counts: [2]' in bad['errors']['pbo']
    assert bad['deflated_sharpe_probability'] is not None
    for file in ('trial_returns.parquet', 'metadata.json'):
        (tmp_path / file).write_bytes((path / file).read_bytes())
    missing = acceptance_report(tmp_path)
    assert missing['pbo'] is None and 'timestamps' in missing['errors']['pbo']
    assert missing['deflated_sharpe_probability'] is not None and missing['pbo_block_bars'] == 12


def test_pbo_configuration_persistence(acceptance_runs, pbo_runs, tmp_path, monkeypatch):
    from limen.cli.commands.resume import run_resume

    (tmp_path / 'inputs').mkdir()
    (tmp_path / 'inputs' / _FIXTURE.name).write_bytes(_FIXTURE.read_bytes())
    monkeypatch.chdir(tmp_path)

    _, config, path = pbo_runs['compiled_only']
    metadata = json.loads((path / 'metadata.json').read_text())
    assert metadata['yaml_reference'] == config
    for acceptance in (None, {'max_pbo': 0.9}):
        report = acceptance_report(path, acceptance=acceptance)
        assert report['n_blocks'] == 8 and report['pbo'] == 6 / 7
        assert report['verdicts'] == ({} if acceptance is None else {'max_pbo': True})
    assert run_resume(path, progress_bar=False)
    assert json.loads((path / 'metadata.json').read_text())['pbo_blocks'] == 8
    _run(pbo_runs['native'][0], resume=True)
    before = _artifact_snapshot(path)
    with pytest.raises(ValueError, match='saved count'):
        acceptance_report(path, acceptance={'pbo_blocks': 2})
    assert _artifact_snapshot(path) == before
    for workers in (1, 2):
        changed = copy.deepcopy(config)
        changed['sfd']['manifest']['acceptance']['pbo_blocks'] = 2
        other = _loop(changed, pl.read_parquet(_FIXTURE), path)
        with pytest.raises(ValueError, match=r'different acceptance\.pbo_blocks'):
            other.run('walk_forward', resume=True, workers=workers, progress_bar=False)
        assert _artifact_snapshot(path) == before
    native_loop, _, native_path = pbo_runs['native']
    native_before = _artifact_snapshot(native_path)
    native_loop.manifest.set_acceptance(pbo_blocks=2)
    with pytest.raises(ValueError, match=r'different acceptance\.pbo_blocks'):
        _run(native_loop, resume=True)
    assert _artifact_snapshot(native_path) == native_before
    native_loop.manifest.set_acceptance(pbo_blocks=8)
    for invalid in (3, True, '8', 8.0):
        corrupted = {**metadata, 'pbo_blocks': invalid}
        (path / 'metadata.json').write_text(json.dumps(corrupted))
        snapshot = _artifact_snapshot(path)
        with pytest.raises(ValueError, match='pbo_blocks'):
            acceptance_report(path)
        assert _artifact_snapshot(path) == snapshot
    (path / 'metadata.json').write_text(json.dumps({**metadata, 'pbo_blocks': 10}))
    snapshot = _artifact_snapshot(path)
    with pytest.raises(ValueError, match='Saved pbo_blocks conflicts'):
        acceptance_report(path)
    assert _artifact_snapshot(path) == snapshot
    corrupted = copy.deepcopy(metadata)
    corrupted['yaml_reference']['sfd']['manifest']['acceptance']['pbo_blocks'] = 2
    (path / 'metadata.json').write_text(json.dumps(corrupted))
    snapshot = _artifact_snapshot(path)
    with pytest.raises(ValueError, match='Saved pbo_blocks conflicts'):
        acceptance_report(path)
    assert _artifact_snapshot(path) == snapshot
    omitted_yaml_count = copy.deepcopy(metadata)
    omitted_yaml_count['yaml_reference']['sfd']['manifest']['acceptance'].pop('pbo_blocks')
    (path / 'metadata.json').write_text(json.dumps(omitted_yaml_count))
    snapshot = _artifact_snapshot(path)
    with pytest.raises(ValueError, match='Saved pbo_blocks conflicts'):
        acceptance_report(path)
    assert _artifact_snapshot(path) == snapshot
    omitted_count = copy.deepcopy(metadata)
    omitted_count.pop('pbo_blocks')
    (path / 'metadata.json').write_text(json.dumps(omitted_count))
    snapshot = _artifact_snapshot(path)
    for settings in (None, {'max_pbo': 0.9}):
        with pytest.raises(ValueError, match='Saved pbo_blocks conflicts'):
            acceptance_report(path, acceptance=settings)
        assert _artifact_snapshot(path) == snapshot
    (path / 'metadata.json').write_text(json.dumps(metadata))
    source = acceptance_runs[0][2]
    for file in ('trial_returns.parquet', 'metadata.json'):
        (tmp_path / file).write_bytes((source / file).read_bytes())
    (tmp_path / 'round_data.jsonl').write_text(_storage.text(_storage.round_path(source)))
    legacy = json.loads((tmp_path / 'metadata.json').read_text())
    legacy.pop('pbo_blocks')
    legacy.pop('round_storage')
    (tmp_path / 'metadata.json').write_text(json.dumps(legacy))
    assert acceptance_report(tmp_path)['n_blocks'] == 2
    (tmp_path / 'metadata.json').unlink()
    snapshot = _artifact_snapshot(tmp_path)
    with pytest.raises(ValueError, match='saved count'):
        acceptance_report(tmp_path, acceptance={'pbo_blocks': 8})
    assert _artifact_snapshot(tmp_path) == snapshot
    assert acceptance_report(tmp_path)['n_blocks'] == 2
    legacy['recording'] = dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), False)
    (tmp_path / 'metadata.json').write_text(json.dumps(legacy))
    for settings in ({'pbo_blocks': 2}, {'max_pbo': 0.5}):
        snapshot = _artifact_snapshot(tmp_path)
        with pytest.raises(ValueError, match=r'recording\.alignment'):
            acceptance_report(tmp_path, acceptance=settings)
        assert _artifact_snapshot(tmp_path) == snapshot
    assert acceptance_report(tmp_path)['n_blocks'] == 2
