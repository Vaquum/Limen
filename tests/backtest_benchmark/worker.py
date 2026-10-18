"""A fresh process per framework and scenario, with no competitor Limen imports."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import json
import os
import platform
import resource
import sys
import time

import numpy as np

from tests.backtest_benchmark import THREADS
from tests.backtest_benchmark.adapters import make_adapter
from tests.backtest_benchmark.contract import (
    BatchResult, assert_parity, reference_result, result_summary, save_result,
)

def package_versions():
    packages = ('vaquum-limen', 'numpy', 'pandas', 'polars', 'vectorbt',
                'vectorbt-rust', 'numba', 'llvmlite', 'backtesting', 'backtrader')
    result = {'python': platform.python_version()}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def peak_rss_mib():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 2 if sys.platform == 'darwin' else 1024)


def timing_summary(samples):
    return {'median_ms': float(np.median(samples)),
            'p95_ms': float(np.percentile(samples, 95)), 'samples_ms': samples}


def _measure(adapter):
    started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()
    raw = adapter.execute()
    executed = time.perf_counter_ns()
    common = adapter.normalize(raw)
    summary = result_summary(common)
    finished = time.perf_counter_ns()
    return common, summary, {
        'engine_ms': (executed - started) / 1e6,
        'normalization_ms': (finished - executed) / 1e6,
        'common_outputs_ms': (finished - started) / 1e6,
        'cpu_ms': (time.process_time_ns() - cpu_started) / 1e6,
    }


def _snapshot_measure(calculate, repeats):
    started = time.perf_counter_ns()
    metrics = calculate()
    first_ms = (time.perf_counter_ns() - started) / 1e6
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        calculate()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return {'first_call_ms': first_ms, 'warm': timing_summary(samples),
            'metrics': {name: float(value) if np.isfinite(value) else None
                        for name, value in metrics.items()},
            'scope': 'backtest_snapshot validation, execution and all native bar metrics'}


def _finite_metrics(metrics):
    return {name: _finite_metrics(value) if isinstance(value, dict)
            else value if np.isfinite(value) else None for name, value in metrics.items()}


def run_worker(args):
    with np.load(args.prepared, allow_pickle=False) as saved:
        data = {name: saved[name] for name in saved.files}
    report = {'engine': args.engine, 'status': 'running',
              'python_executable': sys.executable, 'versions': package_versions(),
              'evaluation_rows': len(data['close']) - 1,
              'processed_rows': len(data['close']) - (1 if args.engine == 'limen' else 0),
              'portfolio_count': data['signal'].shape[1] if data['signal'].ndim == 2 else 1,
              'threads': {name: os.environ.get(name) for name in THREADS},
              'numba_cache_dir': os.environ.get('NUMBA_CACHE_DIR'),
              'rss_before_adapter_mib': peak_rss_mib()}
    started = time.perf_counter_ns()
    try:
        adapter = make_adapter(args.engine, data)
    except ImportError as exc:
        report.update(status='unavailable', reason=f'{type(exc).__name__}: {exc}')
        args.worker_output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        return 0
    report['adapter_preparation_ms'] = (time.perf_counter_ns() - started) / 1e6
    report['scope'] = adapter.scope
    report['settings'] = adapter.settings
    if data['signal'].ndim == 2:
        expected = BatchResult({str(name): reference_result(data['close'], data['signal'][:, index])
                                for index, name in enumerate(data['strategy_names'])})
    else:
        expected = reference_result(data['close'], data['signal'])
    common, summary, cold = _measure(adapter)
    report['cold'] = cold
    report['metrics'] = summary
    report['warm'] = {}
    samples = []
    try:
        assert_parity(common, expected)
        for _ in range(args.repeats):
            warm_result, _, timing = _measure(adapter)
            assert_parity(warm_result, expected)
            samples.append(timing)
    except AssertionError as exc:
        report.update(status='parity_failed', reason=str(exc))
    else:
        report['status'] = 'verified'
    for scope in ('engine_ms', 'normalization_ms', 'common_outputs_ms', 'cpu_ms'):
        if samples:
            report['warm'][scope] = timing_summary([sample[scope] for sample in samples])
    report['peak_rss_mib'] = peak_rss_mib()
    if common.equity.shape == expected.equity.shape and np.isfinite(common.equity).all():
        report['max_abs_equity_error'] = float(np.max(np.abs(common.equity - expected.equity)))
        report['max_relative_equity_error'] = float(np.max(
            np.abs(common.equity - expected.equity) / expected.equity))
    else:
        report['max_abs_equity_error'] = None
        report['max_relative_equity_error'] = None
    if report['status'] == 'parity_failed':
        report['metrics'] = _finite_metrics(summary)
    save_result(args.worker_output.with_suffix('.npz'), common)
    if adapter.snapshot is not None:
        report['native_snapshot'] = _snapshot_measure(adapter.snapshot, args.repeats)
        report['peak_rss_including_snapshot_mib'] = peak_rss_mib()
    report['limen_imported'] = 'limen' in sys.modules
    if args.engine != 'limen' and report['limen_imported']:
        raise AssertionError('Competitor process imported Limen')
    args.worker_output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return 0 if report['status'] == 'verified' else 1
