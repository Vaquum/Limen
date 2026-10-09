import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from limen import Log
from limen.experiment import MLManifest
from limen.experiment import experiment_core
from limen.experiment.errors import StrictModeError
from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import GridStrategy
from limen.targets import ThresholdBinaryTarget


def _observed_model(data):
    labels = np.asarray(data['y_test'])
    return {'positive_fraction': float(labels.mean()), '_preds': labels.tolist()}


def _make_uel(path, msq, counts, *, ablation=True, seeds=(42,)):
    data = pl.read_parquet(Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet').head(500)
    manifest = (MLManifest()
        .set_split_config(3, 1, 1)
        .add_indicator(lambda df: df.with_columns(pl.col('close').pct_change().alias('roc')))
        .add_indicator(lambda df: df.with_columns(pl.col('close').rolling_std(5).alias('vol_5')))
        .add_indicator(lambda df: df.with_columns(pl.col('close').rolling_mean(10).alias('sma_10')))
        .with_target_label('outcome', ThresholdBinaryTarget, fit_params={'source_column': 'roc', 'threshold': 0.0})
        .with_reference_architecture(_observed_model))
    if ablation:
        manifest.set_feature_ablation()
    params = {'feature_drop_count': list(counts), 'feature_drop_seed': list(seeds)}
    sfd = SimpleNamespace(params=lambda: params, manifest=lambda: manifest)
    strategy = GridStrategy(ParamDomain(params)) if msq else None
    uel = UniversalExperimentLoop(data=data, sfd=sfd, search_strategy=strategy,
                                 experiment_dir=path, feedback_interval=1, checkpoint_interval=1)
    seen = []
    original_prep = uel.prep

    def capture_prep(data, round_params):
        result = original_prep(data, round_params)
        dropped = round_params.get('_dropped_features', [])
        assert not set(dropped).intersection(result['_feature_names'])
        seen.append(list(dropped))
        return result

    uel.prep = capture_prep
    return uel, seen


def _run(uel, n, *, post_processing=True, resume=False):
    uel.run('results', n_permutations=n, prep_each_round=True, random_search=False,
            post_processing=post_processing, progress_bar=False, resume=resume)


def _csv_rows(path):
    with (path / 'results.csv').open(newline='') as file:
        return list(csv.DictReader(file))


@pytest.mark.parametrize('msq', (False, True))
@pytest.mark.parametrize('counts', ((0, 1, 2), (2, 0, 1), (0,)))
def test_ablation_rows_round_trip(tmp_path, monkeypatch, msq, counts):
    monkeypatch.setattr(experiment_core, 'STANDARD_RUN_LOG_BATCH_SIZE', 1)
    uel, seen = _make_uel(tmp_path, msq, counts)
    _run(uel, len(counts))
    rows = _csv_rows(tmp_path)
    assert [json.loads(row['_dropped_features']) for row in rows] == seen
    assert sorted(map(len, seen)) == sorted(counts)
    values = [row['_dropped_features'] for row in rows]
    assert uel.experiment_log['_dropped_features'].to_list() == values
    assert uel.experiment_log.schema['_dropped_features'] == pl.String
    for log in (Log(file_path=str(tmp_path / 'results.csv')), Log(uel_object=uel), uel._log):
        assert log.experiment_log['_dropped_features'].tolist() == values
        for feature in ('roc', 'vol_5', 'vol', 'sma_10'):
            membership = log.experiment_log['_dropped_features'].apply(lambda value: feature in json.loads(value))
            assert membership.tolist() == [feature in dropped for dropped in seen]
    uel.experiment_log.write_parquet(tmp_path / 'results.parquet')
    assert pl.read_parquet(tmp_path / 'results.parquet')['_dropped_features'].to_list() == values
    if msq:
        records = [json.loads(line) for line in (tmp_path / 'round_data.jsonl').read_text().splitlines()]
        by_id = {record['round_id']: record['round_params'].get('_dropped_features', []) for record in records}
        assert [by_id[row['id']] for row in rows] == seen
        assert [params.get('_dropped_features', []) for params in uel.round_params] == seen


@pytest.mark.parametrize('msq', (False, True))
def test_non_ablation_output_unchanged(tmp_path, msq):
    uel, _ = _make_uel(tmp_path, msq, (0, 1), ablation=False)
    _run(uel, 2, post_processing=False)
    assert '_dropped_features' not in _csv_rows(tmp_path)[0]
    assert '_dropped_features' not in uel.experiment_log.columns


@pytest.mark.parametrize('msq', (False, True))
@pytest.mark.parametrize('counts', ((0, 1, 2), (1, 0, 2)))
def test_ablation_failed_rows_keep_metadata(tmp_path, monkeypatch, msq, counts):
    monkeypatch.setattr(experiment_core, 'STANDARD_RUN_LOG_BATCH_SIZE', 1)
    uel, seen = _make_uel(tmp_path, msq, counts)
    original_model = uel.model

    def fail_first(data, round_params):
        if len(seen) == 1:
            raise StrictModeError('Recorded ablation round failed after preparation')
        return original_model(data, round_params)

    uel.model = fail_first
    _run(uel, 3, post_processing=False)
    rows = _csv_rows(tmp_path)
    assert rows[0]['strict_mode_error']
    assert [json.loads(row['_dropped_features']) for row in rows] == seen
    assert uel.experiment_log['_dropped_features'].to_list() == [row['_dropped_features'] for row in rows]


@pytest.mark.parametrize('legacy', (False, True))
def test_ablation_resume_preserves_json_or_rejects_legacy(tmp_path, legacy):
    uel, first_seen = _make_uel(tmp_path, True, (0, 1, 2), seeds=(41, 42))
    original_model = uel.model

    def stop_after_two(data, round_params):
        result = original_model(data, round_params)
        if len(first_seen) == 2:
            uel._shutdown_requested = True
        return result

    uel.model = stop_after_two
    _run(uel, 6, post_processing=False)
    assert len(_csv_rows(tmp_path)) == 2
    if legacy:
        rows = _csv_rows(tmp_path)
        for row in rows:
            del row['_dropped_features']
        with (tmp_path / 'results.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        artifacts = {path: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    resumed, next_seen = _make_uel(tmp_path, True, (0, 1, 2), seeds=(41, 42))
    if legacy:
        with pytest.raises(ValueError, match='Cannot resume ablation results without _dropped_features'):
            _run(resumed, 6, resume=True)
        assert all(path.read_bytes() == content for path, content in artifacts.items())
        assert next_seen == []
    else:
        _run(resumed, 6, resume=True)
        rows = _csv_rows(tmp_path)
        assert len(rows) == 6
        assert len({row['id'] for row in rows}) == 6
        assert [json.loads(row['_dropped_features']) for row in rows] == first_seen + next_seen
        assert resumed.experiment_log['_dropped_features'].to_list() == [row['_dropped_features'] for row in rows]
        assert resumed._log.experiment_log['_dropped_features'].tolist() == [row['_dropped_features'] for row in rows]
