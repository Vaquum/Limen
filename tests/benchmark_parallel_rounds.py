import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
_LIMITS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'POLARS_MAX_THREADS')


def _timed_round(function, params, destination):
    started = time.monotonic()
    result = function(params)
    path = Path(destination) / f"{params['_round_index']}.json"
    path.write_text(json.dumps({
        'index': params['_round_index'], 'pid': os.getpid(),
        'started': started, 'finished': time.monotonic(),
        'process_peak_rss': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }))
    return result


class _TimedPool(ProcessPoolExecutor):
    destination = None

    def submit(self, function, params):
        return super().submit(_timed_round, function, params, str(self.destination))


class _MemorySampler:
    def __init__(self):
        self.finished = threading.Event()
        self.peak_bytes = 0
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.finished.is_set():
            output = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss='], text=True)
            rows = [tuple(int(value) for value in line.split()) for line in output.splitlines() if line.strip()]
            descendants = {os.getpid()}
            previous = set()
            while descendants != previous:
                previous = set(descendants)
                descendants.update(pid for pid, parent, _ in rows if parent in descendants)
            rss = sum(kib for pid, _, kib in rows if pid in descendants) * 1024
            self.peak_bytes = max(self.peak_bytes, rss)
            self.finished.wait(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finished.set()
        self.thread.join()


def _configuration():
    from tests.test_parallel_rounds import FIXTURE
    from tests.test_record_model_outputs import _config

    config = _config('lightgbm')
    config['metadata']['name'] = 'parallel_cpu_benchmark'
    manifest = config['sfd']['manifest']
    manifest['data_source'] = {
        'method': 'limen.data.HistoricalData.get_any_file',
        'params': {'file_path_or_url': str(FIXTURE)},
    }
    manifest['split_dates'] = {
        'train_start': '2024-01-01', 'train_end': '2024-08-31',
        'val_start': '2024-09-01', 'val_end': '2024-09-30',
        'test_start': '2024-10-01', 'test_end': '2024-10-31',
    }
    manifest['features'][0]['params']['end'] = 96
    config['sfd']['params'] = {
        'learning_rate': [0.01 + index * 0.005 for index in range(16)],
        'n_estimators': [1200], 'num_leaves': [31], 'max_depth': [8],
        'early_stopping_rounds': [0], 'random_state': [42],
        'n_jobs': [1], 'verbosity': [-1], 'deterministic': [True],
        'force_row_wise': [True],
    }
    config['uel'].update(n_permutations=16, prep_each_round=True,
                         search_strategy={'type': 'grid', 'seed': 43},
                         checkpoint_interval=1000, feedback_interval=1000,
                         record_execution=True, record_model_outputs=True)
    return config


def _overlap(intervals):
    points = [(row['started'], 1) for row in intervals]
    points.extend((row['finished'], -1) for row in intervals)
    current, maximum = 0, 0
    for _, delta in sorted(points):
        current += delta
        maximum = max(maximum, current)
    return maximum


def _campaign(config, directory, workers):
    import polars as pl

    from limen.experiment import UniversalExperimentLoop
    from limen.yaml import CompiledSFD, build_search_strategy
    from tests.test_parallel_rounds import FIXTURE

    markers = directory.parent / f'{directory.name}_timing'
    markers.mkdir()
    _TimedPool.destination = markers
    with _MemorySampler() as memory:
        started = time.monotonic()
        data = pl.read_parquet(FIXTURE)
        interval = (data['datetime'][1] - data['datetime'][0]).total_seconds()
        data = data.with_columns(pl.lit(interval).alias('base_interval'))
        loop = UniversalExperimentLoop(
            sfd=CompiledSFD(config), data=data, experiment_dir=directory,
            yaml_reference=config, search_strategy=build_search_strategy(config),
            feedback_interval=1000, checkpoint_interval=1000,
        )
        with patch('limen.experiment._parallel_rounds.ProcessPoolExecutor', _TimedPool):
            loop.run(config['metadata']['name'], n_permutations=16,
                     prep_each_round=True, progress_bar=False, workers=workers,
                     record_execution=True, record_model_outputs=True)
        elapsed = time.monotonic() - started
    intervals = [json.loads(path.read_text()) for path in markers.glob('*.json')]
    return {
        'workers': workers, 'rounds': loop.experiment_log.height,
        'elapsed_seconds': elapsed,
        'rounds_per_second': loop.experiment_log.height / elapsed,
        'peak_process_tree_rss_bytes': memory.peak_bytes,
        'maximum_evaluation_overlap': _overlap(intervals) if intervals else 1,
        'worker_intervals': sorted(intervals, key=lambda row: row['index']),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    destination = parser.parse_args().output
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    if cpus is None or cpus < 4:
        raise RuntimeError('The recorded four-worker benchmark requires four available CPUs')
    for name in _LIMITS:
        os.environ[name] = '1'
    from limen import __version__
    from tests.test_parallel_rounds import FIXTURE, assert_runs_equal

    destination.mkdir(parents=True, exist_ok=False)
    config = _configuration()
    serial = _campaign(config, destination / 'serial', 1)
    parallel = _campaign(config, destination / 'parallel', 4)
    assert_runs_equal(destination / 'parallel', destination / 'serial')
    speedup = serial['elapsed_seconds'] / parallel['elapsed_seconds']
    report = {
        'limen_version': __version__, 'python_version': sys.version,
        'platform': platform.platform(), 'available_cpus': cpus,
        'fixture_sha256': hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        'configuration': config, 'serial': serial, 'parallel': parallel,
        'speedup': speedup, 'target_speedup': 3.0,
        'target_met': speedup >= 3.0,
        'thread_limits': {name: os.environ[name] for name in _LIMITS},
    }
    path = destination / 'benchmark.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False))
    sys.stdout.write(json.dumps({'report': str(path), 'speedup': speedup,
                                 'target_met': report['target_met'],
                                 'maximum_evaluation_overlap': parallel['maximum_evaluation_overlap']}) + '\n')


if __name__ == '__main__':
    main()
