import csv
import json
import shutil
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from threadpoolctl import threadpool_limits

from limen.experiment import UniversalExperimentLoop
from limen.yaml import CompiledSFD, build_search_strategy

from tests.test_record_execution import _config as _binary_config

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('confusion_tp_mean_return_pct', 'confusion_fp_mean_return_pct',
           'backtest_avg_win_bps', 'backtest_avg_loss_bps')
RECORDING = dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True)


def _config():
    config = _binary_config('binary')
    config['sfd']['manifest']['calibration'] = {
        'probability_calibration': {'func': 'limen.calibration.sklearn_probability_calibrator',
                                    'params': {'method': 'sigmoid'}},
        'threshold_function': {'func': 'limen.calibration.grid_threshold_optimizer',
                               'params': {'metric': 'limen.metrics.balanced_metric.balanced_metric',
                                          'threshold_min': 0.4, 'threshold_max': 0.6, 'threshold_step': 0.05}},
    }
    config['sfd']['params'] = {
        'C': [1e-8, 1.0], 'max_iter': [1000], 'n_jobs': [1],
        'use_calibration': [False], 'use_threshold': [False, True],
    }
    config['uel'].update(n_permutations=4, feedback_interval=1)
    return config


def _loop(config, bars, path, *, callback=None):
    return UniversalExperimentLoop(
        sfd=CompiledSFD(config), data=bars, experiment_dir=path, yaml_reference=config,
        search_strategy=build_search_strategy(config), checkpoint_interval=1,
        feedback_interval=1, intra_callback=callback,
    )


def _run(loop, *, workers=1, resume=False):
    with threadpool_limits(limits=1):
        loop.run('resume_metric_types', n_permutations=4, prep_each_round=True, progress_bar=False,
                 workers=workers, resume=resume, post_processing=workers == 1,
                 record_execution=True, record_model_outputs=True, recording=RECORDING)


def _decoded_rows(path):
    with (path / 'results.csv').open(newline='') as stream:
        return [{key: value for key, value in row.items() if key != 'execution_time'}
                for row in csv.DictReader(stream)]


def _comparable(loop, control):
    expected = control.experiment_log.drop('execution_time')
    return loop.experiment_log.drop('execution_time').select(expected.columns), expected


@pytest.fixture(scope='module')
def recorded_bars():
    return pl.read_parquet(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet').head(1200)


@pytest.fixture(scope='module')
def control(recorded_bars, tmp_path_factory):
    config = _config()
    path = tmp_path_factory.mktemp('metric-types-control')
    loop = _loop(config, recorded_bars, path)
    _run(loop)
    assert loop.experiment_log.height == 4
    for metric in METRICS:
        assert loop.experiment_log.schema[metric] == pl.Float64
        assert loop.experiment_log[metric].is_nan().any()
    assert loop.experiment_log.schema['optimal_threshold'] == pl.Float64
    assert loop.experiment_log['optimal_threshold'].null_count() == 2
    assert any(row[metric] == 'nan' for row in _decoded_rows(path) for metric in METRICS)
    return {'config': config, 'loop': loop, 'path': path}


@pytest.mark.parametrize('workers', (1, 2))
def test_resume_preserves_numeric_metric_types(workers, control, recorded_bars, tmp_path):
    config = control['config']
    path = tmp_path / 'interrupted'
    holder = {}

    def stop_after_two(log, _msq):
        if log.height == 2:
            holder['loop']._shutdown_requested = True

    first = _loop(config, recorded_bars, path, callback=stop_after_two)
    holder['loop'] = first
    _run(first, workers=workers)
    checkpoint = json.loads((path / 'checkpoint.json').read_text())
    assert checkpoint['metadata']['experiment_round'] + 1 == 2
    prefix = _decoded_rows(path)
    assert len(prefix) == 2
    assert any(row[metric] == 'nan' for row in prefix for metric in METRICS)
    assert all(row['optimal_threshold'] == '' for row in prefix)

    observed = []

    def observe(log, _msq):
        observed.append({metric: log.schema[metric] for metric in METRICS})

    resumed = _loop(config, recorded_bars, path, callback=observe)
    _run(resumed, workers=workers, resume=True)
    actual, expected = _comparable(resumed, control['loop'])
    assert_frame_equal(actual, expected, check_exact=True)
    assert _decoded_rows(path) == _decoded_rows(control['path'])
    assert observed and all(dtype == pl.Float64 for seen in observed for dtype in seen.values())


@pytest.mark.parametrize('workers', (1, 2))
def test_completed_resume_preserves_numeric_metric_types(workers, control, recorded_bars, tmp_path):
    source = control['path']
    if workers == 2:
        source = tmp_path / 'parallel-source'
        completed = _loop(control['config'], recorded_bars, source)
        _run(completed, workers=2)
        assert completed.experiment_log.height == 4
    path = tmp_path / 'completed'
    shutil.copytree(source, path)
    loop = _loop(control['config'], recorded_bars, path)
    _run(loop, workers=workers, resume=True)
    actual, expected = _comparable(loop, control['loop'])
    assert_frame_equal(actual, expected, check_exact=True)
    assert _decoded_rows(path) == _decoded_rows(control['path'])
