"""Bounded recorded-data comparison of fitted preparation reuse in factorization."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet'
_LIMITS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'POLARS_MAX_THREADS')
_CASES = ('fresh_fitted_preparation', 'reuse_fitted_preparation')


def _configuration():
    from limen import __version__

    return {
        'schema_version': '1.0',
        'metadata': {'name': 'factorized_preparation_benchmark',
                     'limen_version': __version__, 'mode': 'development',
                     'description': 'Recorded fitted-preparation comparison within factorize=True'},
        'sfd': {
            'manifest': {
                'type': 'ml',
                'data_source': {'method': 'limen.data.HistoricalData.get_any_file',
                                'params': {'file_path_or_url': str(FIXTURE)}},
                'split_dates': {
                    'train_start': '2024-01-01', 'train_end': '2024-08-31',
                    'val_start': '2024-09-01', 'val_end': '2024-09-30',
                    'test_start': '2024-10-01', 'test_end': '2024-10-31',
                },
                'indicators': [{'func': 'limen.indicators.window_return', 'params': {'period': 1}}],
                'features': [{'func': 'limen.features.lag_range',
                              'params': {'col': 'ret_1', 'start': 0, 'end': 192}}],
                'target': {'name': 'up_next', 'class': 'limen.targets.QuantileBinaryTarget',
                           'fit_params': {'source_column': 'ret_1', 'quantile': 0.5},
                           'transform_params': {'shift': -1}},
                'scaler': {'class': 'limen.scalers.RobustScaler'},
                'strict_mode': True,
                'reference_architecture': 'limen.sfd.reference_architecture.lightgbm_binary.lightgbm_binary',
                'backtest': {'fee_bps': '{fee}', 'slip_bps': 2.0, 'notional_rate': 1.0},
            },
            'params': {
                'learning_rate': [0.025 + index * 0.005 for index in range(24)],
                'n_estimators': [8], 'num_leaves': [7], 'max_depth': [4],
                'early_stopping_rounds': [0], 'random_state': [42], 'n_jobs': [1],
                'verbosity': [-1], 'deterministic': [True], 'force_row_wise': [True],
                'boosting_type': ['gbdt'], 'subsample_freq': [0], 'subsample': [1.0],
                'colsample_bytree': [1.0], 'fee': [3.0, 12.0],
            },
        },
        'uel': {'n_permutations': 48, 'search_strategy': {'type': 'grid'},
                'factorize': True, 'prep_each_round': True,
                'checkpoint_interval': 1000, 'feedback_interval': 1000,
                'record_execution': True, 'record_model_outputs': True},
    }


def _timed_call(function, stages, name):
    stages[name] = {'calls': 0, 'seconds': 0.0}

    @wraps(function)
    def measured(*args, **kwargs):
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            stages[name]['calls'] += 1
            stages[name]['seconds'] += time.perf_counter() - started

    return measured


def _captured(directory):
    with (directory / 'results.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    return {
        'columns': list(rows[0]), 'rows': rows,
        'round_data': [json.loads(line) for line in (directory / 'round_data.jsonl').read_text().splitlines()],
        'metadata': json.loads((directory / 'metadata.json').read_text()),
    }


def _campaign(case, directory, report_path):
    # Imports, source loading, construction, IPC and writes are inside parent timing.
    import lightgbm
    import numpy as np
    import polars as pl
    import sklearn
    from threadpoolctl import threadpool_info, threadpool_limits

    from limen import __version__
    from limen.experiment import UniversalExperimentLoop
    from limen.experiment._factorize import FactorizedRounds
    from limen.experiment.manifest_core import MLManifest
    from limen.scalers import RobustScaler
    from limen.sfd.reference_architecture.lightgbm_binary import LightGBMBinary
    from limen.targets import QuantileBinaryTarget
    from limen.yaml import CompiledSFD, build_search_strategy, validate
    from tests.fixtures.factorize.generate import component_digests, digest

    config = _configuration()
    validation = validate(config)
    if not validation.valid:
        raise ValueError(validation.errors)
    stages = {}
    original_evaluate = FactorizedRounds.evaluate

    def without_fitted_reuse(rounds, params):
        rounds.preparation.cache.clear()
        return original_evaluate(rounds, params)

    started = time.perf_counter()
    data = pl.read_parquet(FIXTURE)
    loop = UniversalExperimentLoop(
        sfd=CompiledSFD(config), data=data, experiment_dir=directory,
        yaml_reference=config, search_strategy=build_search_strategy(config),
        feedback_interval=1000, checkpoint_interval=1000,
    )
    with ExitStack() as stack:
        stack.enter_context(threadpool_limits(limits=1))
        for owner, attribute, name in (
            (QuantileBinaryTarget, '__init__', 'target_fit'),
            (RobustScaler, '__init__', 'scaler_fit'),
            (MLManifest, '_prepare_data_with_fitted', 'preparation'),
            (LightGBMBinary, 'train', 'model_fit'),
            (LightGBMBinary, 'predict', 'model_prediction'),
        ):
            stack.enter_context(patch.object(owner, attribute,
                                           _timed_call(getattr(owner, attribute), stages, name)))
        if case == _CASES[0]:
            stack.enter_context(patch.object(FactorizedRounds, 'evaluate', without_fitted_reuse))
        loop.run(config['metadata']['name'], n_permutations=48, prep_each_round=True,
                 progress_bar=False, post_processing=False, factorize=True, workers=1,
                 record_execution=True, record_model_outputs=True)
        native_pools = threadpool_info()
        if pl.thread_pool_size() != 1 or any(pool['num_threads'] != 1 for pool in native_pools):
            raise AssertionError(f'Expected one native thread: {native_pools}')
    elapsed = time.perf_counter() - started
    captured = _captured(directory)
    scientific_digest = digest(captured)
    components = component_digests(captured)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        'case': case, 'limen_version': __version__, 'python': sys.version,
        'platform': platform.platform(),
        'versions': {'numpy': np.__version__, 'polars': pl.__version__,
                     'sklearn': sklearn.__version__, 'lightgbm': lightgbm.__version__},
        'rounds': len(captured['rows']), 'stages': stages,
        'campaign_elapsed_seconds': elapsed,
        'peak_process_rss_bytes': int(peak if sys.platform == 'darwin' else peak * 1024),
        'scientific_digest_12dp': scientific_digest,
        'component_digests_12dp': components,
        'native_pools': native_pools,
        'configuration': config,
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    sys.stdout.write(json.dumps({'case': case, 'seconds': elapsed, 'stages': stages}) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=_CASES)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.case is not None:
        if args.report is None:
            parser.error('--case requires --report')
        _campaign(args.case, args.output, args.report)
        return

    args.output.mkdir(parents=True, exist_ok=False)
    environment = {**os.environ, **{name: '1' for name in _LIMITS}}
    campaigns = {}
    benchmark_started = time.perf_counter()
    for case in _CASES:
        report_path = args.output / f'{case}.json'
        started = time.perf_counter()
        subprocess.run(
            [sys.executable, '-m', 'tests.benchmark_factorized_preparation',
             '--case', case, '--output', str(args.output / case), '--report', str(report_path)],
            cwd=ROOT, env=environment, check=True, timeout=240,
        )
        end_to_end = time.perf_counter() - started
        campaigns[case] = json.loads(report_path.read_text())
        campaigns[case]['end_to_end_seconds'] = end_to_end
        campaigns[case]['rounds_per_second'] = campaigns[case]['rounds'] / end_to_end
    baseline, reuse = (campaigns[case] for case in _CASES)
    if baseline['scientific_digest_12dp'] != reuse['scientific_digest_12dp']:
        differing = [key for key, value in baseline['component_digests_12dp'].items()
                     if reuse['component_digests_12dp'].get(key) != value]
        raise AssertionError(f'Scientific output changed: {differing}')
    for case, fits in ((baseline, 24), (reuse, 1)):
        if case['rounds'] != 48:
            raise AssertionError(f"Expected 48 terminal rows, found {case['rounds']}")
        for name in ('target_fit', 'scaler_fit'):
            if case['stages'][name]['calls'] != fits:
                raise AssertionError(f'{name} expected {fits} fits: {case["stages"]}')
        if case['stages']['model_fit']['calls'] != 24:
            raise AssertionError(f'Prediction reuse changed: {case["stages"]}')
    speedup = baseline['end_to_end_seconds'] / reuse['end_to_end_seconds']
    report = {
        'fixture_sha256': hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        'comparison': 'Same current implementation with factorize=True; baseline disables only fitted-preparation reuse',
        'thread_limits': {name: '1' for name in _LIMITS}, 'workers': 1,
        'campaigns': campaigns,
        'end_to_end_speedup': speedup, 'faster_on_this_workload': speedup > 1.0,
        'total_elapsed_seconds': time.perf_counter() - benchmark_started,
        'scientific_outputs_equal_at_12dp': True,
    }
    path = args.output / 'benchmark.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False))
    sys.stdout.write(json.dumps({'report': str(path), 'speedup': speedup,
                                'equal_at_12dp': True}) + '\n')


if __name__ == '__main__':
    main()
