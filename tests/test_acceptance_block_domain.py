import io
import json
import math
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from ruamel.yaml import YAML

from limen.cli.commands.resume import run_resume
from limen.experiment import UniversalExperimentLoop
from limen.experiment._acceptance_settings import partition_geometry
from limen.experiment.acceptance_report import acceptance_report
from limen.experiment.family_acceptance_report import family_acceptance_report
from limen.metrics import probability_of_backtest_overfitting
from limen.yaml import build_search_strategy
from limen.yaml.compiler import build_manifest
from limen.yaml.store import canonical_manifest_id, commit_manifest
from tests.test_acceptance_gates import _artifact_snapshot, _reference_pbo
from tests.test_record_model_outputs import _config as _ml_config
from tests.test_walk_forward_uel import _config as _rule_config
from tests.test_walk_forward_uel import _loop, _run


_FIXTURE = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'
_ROUTES = ((10, 'rule', True), (12, 'rule', False),
           (10, 'ml', False), (12, 'ml', True))


def _native_loop(history, directory, manifest):
    sfd = SimpleNamespace(__name__=__name__, params=lambda: history.config['sfd']['params'],
                          manifest=lambda: manifest)
    return UniversalExperimentLoop(
        sfd=sfd, data=history.bars, experiment_dir=directory,
        search_strategy=build_search_strategy(history.config),
        yaml_reference=history.config, checkpoint_interval=1,
    )


def _new_loop(history):
    if history.compiled:
        return _loop(history.config, history.bars, history.directory)
    return _native_loop(history, history.directory, build_manifest(history.config))


def _history(project, count, kind='rule', compiled=True, *, test_bars=61, n_permutations=2):
    project.mkdir(parents=True, exist_ok=True)
    config = _rule_config() if kind == 'rule' else _ml_config()
    manifest = config['sfd']['manifest']
    manifest.pop('split_dates', None)
    manifest['split_walk_forward'] = {
        'n_folds': 2, 'test_bars': test_bars, 'purge_bars': 4,
        'embargo_bars': 3, 'anchored': True,
    }
    manifest['acceptance'] = {
        'pbo_blocks': count, 'max_pbo': 1.0, 'min_deflated_sharpe_probability': 0.0,
    }
    manifest.setdefault('backtest', {'fee_bps': 7.0, 'slip_bps': 3.0, 'notional_rate': 0.5})['product'] = {
        'kind': 'cash_spot', 'instrument': 'BTCUSDT', 'base_currency': 'BTC',
        'quote_currency': 'USDT', 'quantity_step': 1e-9, 'min_notional': 0.0,
    }
    bars = pl.read_parquet(_FIXTURE)
    if test_bars == 61:
        bars = bars.head(384)
    bars = bars.with_columns(pl.lit(3600).alias('base_interval'))
    input_path = project / 'input.parquet'
    bars.write_parquet(input_path)
    manifest['data_source'] = {
        'method': 'limen.data.HistoricalData.get_any_file',
        'params': {'file_path_or_url': str(input_path)},
    }
    config['uel'].update(n_permutations=n_permutations, checkpoint_interval=1)
    yaml_path = project / 'experiment.yaml'
    with yaml_path.open('w') as stream:
        YAML().dump(config, stream)
    commit_manifest(yaml_path, project)
    history = SimpleNamespace(config=config, bars=bars, directory=project / 'run',
                              project=project, count=count, kind=kind, compiled=compiled)
    _run(_new_loop(history), n_permutations=n_permutations)
    return history


def _report(history):
    return json.loads((history.directory / 'acceptance_report.json').read_text())


def _matrix(history):
    frame = pl.read_parquet(history.directory / 'trial_returns.parquet')
    return np.stack([group.sort('bar')['net_return'].to_numpy()
                     for group in frame.partition_by('trial', maintain_order=True)])


@pytest.fixture(scope='module', params=_ROUTES,
                ids=('rule-compiled-10', 'rule-native-12', 'ml-native-10', 'ml-compiled-12'))
def recorded_partition(request, tmp_path_factory):
    count, kind, compiled = request.param
    project = tmp_path_factory.mktemp(f'pbo-{count}-{kind}-{"compiled" if compiled else "native"}')
    return _history(project, count, kind, compiled)


def test_recorded_block_count_lifecycle(recorded_partition):
    history = recorded_partition
    directory = history.directory
    initial = _report(history)
    matrix = _matrix(history)
    assert matrix.shape == (2, 120)
    assert initial['n_blocks'] == history.count
    assert initial['pbo_block_bars'] == 120 // history.count
    assert initial['pbo_combinations'] == math.comb(history.count, history.count // 2)
    assert initial['pbo'] is not None
    assert initial['deflated_sharpe_probability'] is not None
    metadata = json.loads((directory / 'metadata.json').read_text())
    assert metadata['pbo_blocks'] == history.count
    assert metadata['yaml_reference']['sfd']['manifest']['acceptance']['pbo_blocks'] == history.count
    assert canonical_manifest_id(metadata['yaml_reference']) == canonical_manifest_id(history.config)

    for settings in (None, {'max_pbo': 1.0}):
        regenerated = acceptance_report(directory, acceptance=settings)
        assert regenerated['n_blocks'] == history.count
        assert regenerated['pbo'] == initial['pbo']
        assert regenerated['deflated_sharpe_probability'] == initial['deflated_sharpe_probability']

    if history.compiled and history.kind == 'rule':
        assert run_resume(directory, progress_bar=False)
    else:
        _run(_new_loop(history), resume=True)
    resumed = _report(history)
    assert resumed['n_blocks'] == history.count
    assert resumed['pbo'] == initial['pbo']
    np.testing.assert_array_equal(_matrix(history), matrix)
    assert 'Nominal counting step (float approximation)' in (directory / 'acceptance_report.md').read_text()


def test_recorded_block_count_parity(recorded_partition):
    history = recorded_partition
    matrix = _matrix(history)
    expected = _reference_pbo(matrix, history.count)
    assert probability_of_backtest_overfitting(matrix, n_blocks=history.count) == expected
    assert _report(history)['pbo'] == expected
    before = _artifact_snapshot(history.project)
    comparison = family_acceptance_report(
        [history.directory], project_root=history.project, n_blocks=history.count,
    )['comparisons'][0]
    assert comparison['n_trials'] == matrix.shape[0]
    assert comparison['n_bars'] == matrix.shape[1]
    assert comparison['pbo'] == expected
    assert _artifact_snapshot(history.project) == before


def test_block_count_evidence_and_provenance_refusals(recorded_partition, tmp_path):
    history = recorded_partition
    before = _artifact_snapshot(history.project)
    with pytest.raises(ValueError, match='conflicts with the saved count'):
        acceptance_report(history.directory, acceptance={'pbo_blocks': history.count + 2})
    assert _artifact_snapshot(history.project) == before

    changed = deepcopy(history.config)
    changed['sfd']['manifest']['acceptance']['pbo_blocks'] = history.count + 2
    with pytest.raises(ValueError, match='Cannot resume'):
        _run(_loop(changed, history.bars, history.directory), resume=True)
    assert _artifact_snapshot(history.project) == before
    missing = tmp_path / 'missing-timestamps'
    missing.mkdir()
    for name in ('trial_returns.parquet', 'metadata.json'):
        shutil.copyfile(history.directory / name, missing / name)
    report = acceptance_report(missing, acceptance={'max_pbo': 1.0})
    assert report['pbo'] is report['verdicts']['max_pbo'] is None
    assert 'timestamps' in report['errors']['pbo']
    assert report['deflated_sharpe_probability'] == _report(history)['deflated_sharpe_probability']
    assert _artifact_snapshot(history.project) == before
    if history.count == 10 and history.kind == 'rule':
        _assert_decimal_refusals(history, tmp_path)


def _assert_decimal_refusals(history, tmp_path):
    assert sys.get_int_max_str_digits() > 0
    digits = max(5000, sys.get_int_max_str_digits() + 1)
    oversized = 10 ** digits
    manifest = build_manifest(history.config)
    with pytest.raises(ValueError, match='integer string conversion'):
        manifest.set_acceptance(pbo_blocks=oversized)
    manifest.acceptance['pbo_blocks'] = oversized
    rejected = tmp_path / 'rejected'
    with pytest.raises(ValueError, match='integer string conversion'):
        _run(_native_loop(history, rejected, manifest))
    assert not rejected.exists() or not list(rejected.iterdir())

    config = deepcopy(history.config)
    config['sfd']['manifest']['acceptance']['pbo_blocks'] = oversized
    with pytest.raises(ValueError, match='integer string conversion'):
        _run(_loop(config, history.bars, rejected))
    assert not rejected.exists() or not list(rejected.iterdir())
    with pytest.raises(ValueError, match='integer string conversion'):
        canonical_manifest_id(config)

    stream = io.StringIO()
    YAML().dump(history.config, stream)
    original = f'pbo_blocks: {history.count}'
    assert original in stream.getvalue()
    yaml_path = tmp_path / 'unrepresentable.yaml'
    yaml_path.write_text(stream.getvalue().replace(original, 'pbo_blocks: 1' + '0' * digits))
    store_root = tmp_path / 'store'
    store_root.mkdir()
    before = _artifact_snapshot(store_root)
    with pytest.raises(ValueError, match='integer string conversion'):
        commit_manifest(yaml_path, store_root)
    assert _artifact_snapshot(store_root) == before

    saved = tmp_path / 'saved'
    shutil.copytree(history.directory, saved)
    metadata_path = saved / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['pbo_blocks'] = 'unrepresentable'
    metadata_path.write_text(json.dumps(metadata).replace('"unrepresentable"', '1' + '0' * digits))
    before = _artifact_snapshot(saved)
    with pytest.raises(ValueError, match='Malformed JSON') as refusal:
        acceptance_report(saved)
    assert 'integer string conversion' in str(refusal.value.__cause__)
    assert _artifact_snapshot(saved) == before


@pytest.mark.parametrize(('count', 'test_bars', 'retained_bars'),
                         ((10 ** 1000, 61, 120), (1100, 1101, 2200)),
                         ids=('impossible-geometry', 'reciprocal-underflow'))
def test_large_block_count_unavailability(tmp_path, count, test_bars, retained_bars):
    history = _history(tmp_path / 'large', count, test_bars=test_bars, n_permutations=1)
    report = _report(history)
    assert report['n_blocks'] == count
    assert report['n_bars'] == retained_bars
    assert report['n_trials'] == 1
    assert report['pbo'] is None
    assert report['verdicts']['max_pbo'] is None
    assert report['deflated_sharpe_probability'] is not None
    assert report['verdicts']['min_deflated_sharpe_probability'] is True
    assert report['errors']['pbo']
    if count > retained_bars:
        assert report['pbo_block_bars'] is None
        assert report['pbo_combinations'] is None
        assert report['pbo_resolution'] is None
        assert 'equal blocks' in report['errors']['pbo']
    else:
        assert report['pbo_block_bars'] == 2
        assert report['pbo_combinations'] == math.comb(count, count // 2)
        assert report['pbo_resolution'] == 0.0
        text = (history.directory / 'acceptance_report.md').read_text()
        assert 'float approximation' in text
        assert 'underflow' in text
        assert '1 / nominal combinations' in text


def test_nominal_geometry_decimal_representation():
    for n_bars in (None, 120):
        geometry = partition_geometry(n_bars, 10 ** 1000)
        assert geometry == {'pbo_combinations': None, 'pbo_resolution': None, 'pbo_block_bars': None}
        json.dumps(geometry, allow_nan=False)
    geometry = partition_geometry(30000, 15000)
    assert geometry == {'pbo_combinations': None, 'pbo_resolution': None, 'pbo_block_bars': 2}
    json.dumps(geometry, allow_nan=False)
