import copy
import json
import math
from itertools import combinations
from pathlib import Path
from statistics import NormalDist

import numpy as np
import polars as pl
import pytest

from limen.metrics import deflated_sharpe_ratio, probability_of_backtest_overfitting
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
    assert all(before >= after for before, after in zip(probabilities, probabilities[1:], strict=False))
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
    assert canonical['sfd']['manifest']['data_source']['params']['file_path_or_url'] == str(_FIXTURE.relative_to(Path(__file__).parents[1]))
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
    permissive = acceptance_report(tmp_path, acceptance={
        'min_deflated_sharpe_probability': 0.0, 'max_pbo': 1.0,
    })
    assert permissive['verdicts'] == {
        'min_deflated_sharpe_probability': True, 'max_pbo': True,
    }


def test_report_end_to_end(acceptance_runs):
    expected_keys = {
        'n_trials', 'n_bars', 'n_blocks', 'winner_trial', 'winner_sharpe',
        'trial_sharpe_variance', 'deflated_sharpe_probability', 'pbo',
        'thresholds', 'verdicts', 'errors',
    }
    for loop, config, path in acceptance_runs:
        report = json.loads((path / 'acceptance_report.json').read_text())
        assert set(report) == expected_keys
        artifact = pl.read_parquet(path / 'trial_returns.parquet')
        trials = artifact['trial'].unique(maintain_order=True).to_list()
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
