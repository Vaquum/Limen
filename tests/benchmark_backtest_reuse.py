"""Compare immutable source trees on identical recorded backtest inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORKER = ROOT / 'tests/backtest_reuse/worker.py'
SCENARIOS = ('snapshot_ledger', 'snapshot_full', 'line_features',
             'timer_execution', 'funded_execution', 'forward_labels')
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'POLARS_MAX_THREADS')
ATOL = 1e-10
MIN_WARM_SPEEDUP = 1.0


def _code_digest(root):
    if not (root / 'limen/__init__.py').is_file():
        raise ValueError(f'Missing implementation tree: {root}')
    digest = hashlib.sha256()
    for path in sorted(root.glob('limen/**/*.py')):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_array(actual, expected, name):
    import numpy as np

    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f'{name}: array shape or dtype changed')
    if actual.dtype.kind in 'fc' or expected.dtype.kind in 'fc':
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0,
                                   equal_nan=True, err_msg=name)
        finite = np.isfinite(actual) & np.isfinite(expected)
        return float(np.max(np.abs(actual[finite] - expected[finite]), initial=0))
    np.testing.assert_array_equal(actual, expected, err_msg=name)
    return 0.0


def _compare(baseline, optimized):
    import numpy as np
    import polars as pl
    from polars.testing import assert_frame_equal

    errors = {}
    float_values = 0
    with (np.load(baseline / 'arrays.npz', allow_pickle=False) as expected,
          np.load(optimized / 'arrays.npz', allow_pickle=False) as actual):
        if actual.files != expected.files:
            raise AssertionError('Output array names or order changed')
        for name in actual.files:
            errors[f'arrays:{name}'] = _assert_array(actual[name], expected[name], name)
            if actual[name].dtype.kind in 'fc':
                float_values += actual[name].size
    names = sorted(path.name for path in baseline.glob('*.parquet'))
    if names != sorted(path.name for path in optimized.glob('*.parquet')):
        raise AssertionError('Output table names changed')
    for name in names:
        expected, actual = pl.read_parquet(baseline / name), pl.read_parquet(optimized / name)
        assert_frame_equal(actual, expected, check_exact=False, abs_tol=ATOL, rel_tol=0)
        for column, dtype in actual.schema.items():
            if dtype.is_float():
                label = f'{name}:{column}'
                errors[label] = _assert_array(actual[column].to_numpy(),
                                              expected[column].to_numpy(), label)
                float_values += actual.height
    return {'max_absolute_error': max(errors.values()),
            'max_absolute_error_by_output': errors,
            'floating_values_compared': float_values,
            'tables_compared': names}


def _assert_campaign_contract(report, workloads=SCENARIOS):
    if report['equivalent'] is not True:
        raise AssertionError('Recorded benchmark equivalence failed')
    records = report['scenarios']
    if tuple(record['scenario'] for record in records) != tuple(workloads):
        raise AssertionError('Recorded benchmark workloads are missing or out of order')
    for record in records:
        if record['equivalent'] is not True:
            raise AssertionError(f'{record["scenario"]}: numerical equivalence failed')
        error = record['max_absolute_error']
        if not math.isfinite(error) or not 0 <= error <= ATOL:
            raise AssertionError(f'{record["scenario"]}: absolute error exceeds {ATOL}')
        speedup = record['warm_speedup']
        if not math.isfinite(speedup) or speedup <= MIN_WARM_SPEEDUP:
            raise AssertionError(f'{record["scenario"]}: warm speedup must exceed {MIN_WARM_SPEEDUP}')


def _prepare(args):
    import numpy as np

    from limen.sfd.rule_based.common_strategies import strategy_signals
    from tests.backtest_benchmark.prepare import read_bars

    prepared = {}
    sources = {}
    for name, path in (('primary', args.input), ('target', args.target_input)):
        frame = read_bars(path)
        destination = args.output / f'{name}.parquet'
        frame.write_parquet(destination)
        signal_path = args.output / f'{name}-signal.npy'
        np.save(signal_path, strategy_signals(frame, args.strategy), allow_pickle=False)
        prepared[name] = (destination, signal_path)
        sources[name] = {'path': str(path), 'sha256': _file_digest(path),
                         'rows': frame.height,
                         'prepared_sha256': _file_digest(destination),
                         'signal_sha256': _file_digest(signal_path)}
        if name == 'primary':
            timer_signal = args.output / 'timer-signal.npy'
            np.save(timer_signal, strategy_signals(frame, 'buy_and_hold'), allow_pickle=False)
            prepared['timer'] = (destination, timer_signal)
    funding = args.output / 'recorded-funding.json'
    funding.write_bytes(args.funding_input.read_bytes())
    sources['funding'] = {'path': str(args.funding_input), 'sha256': _file_digest(funding)}
    prepared['funding'] = funding
    return prepared, sources


def campaign(args):
    if args.output.exists():
        raise ValueError('Output directory must be new; prior measurements are immutable')
    args.output.mkdir(parents=True)
    roots = {'baseline': args.baseline_root, 'optimized': args.optimized_root}
    digests = {name: _code_digest(root) for name, root in roots.items()}
    prepared, sources = _prepare(args)
    report = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'platform': platform.platform(), 'python': sys.version,
              'source_roots': {name: str(root) for name, root in roots.items()},
              'source_code_sha256': digests, 'worker_sha256': _file_digest(WORKER),
              'sources': sources, 'strategy': args.strategy,
              'timer_strategy': 'buy_and_hold', 'repeats': args.repeats,
              'rounds': args.rounds, 'target_rows': args.target_rows,
              'threads': dict.fromkeys(THREADS, '1'),
              'contract': {'absolute_tolerance': ATOL, 'relative_tolerance': 0,
                           'minimum_warm_speedup_exclusive': MIN_WARM_SPEEDUP,
                           'required_workloads': list(args.scenarios),
                           'non_numeric_outputs': 'exact names, order, schemas, nulls, IDs and clocks',
                           'scope': 'source reads and signal preparation excluded; complete declared computation included',
                           'cold': 'first timed call in a fresh implementation process',
                           'warm': 'median of repeated calls within that process',
                           'cold_and_memory': 'advisory; no universal cold-call or memory regression bound',
                           'memory_scope': 'matched fresh-process high-water RSS, including imports, source loading and output persistence; timings include the materialized declared computation and exclude source reads and output persistence',
                           'rss_units': 'resource.ru_maxrss: bytes on Darwin, KiB on Linux; normalized to bytes and MiB',
                           'p95': 'sample percentile; three repeats do not establish tail latency'},
              'scenarios': []}
    environment = {**os.environ, **dict.fromkeys(THREADS, '1'), 'PYTHONHASHSEED': '0'}
    for scenario in args.scenarios:
        key = 'target' if scenario == 'forward_labels' else (
            'timer' if scenario == 'timer_execution' else 'primary')
        source, signal = prepared[key]
        record = {'scenario': scenario, 'results': {}}
        for name, root in roots.items():
            directory = args.output / scenario / name
            subprocess.run(
                [sys.executable, str(WORKER), '--root', str(root), '--input', str(source),
                 '--signal', str(signal), '--output', str(directory), '--variant', name,
                 '--funding', str(prepared['funding']),
                 '--scenario', scenario, '--repeats', str(args.repeats),
                 '--rounds', str(args.rounds), '--target-rows', str(args.target_rows)],
                cwd=args.output, env=environment, check=True, timeout=240,
            )
            record['results'][name] = json.loads((directory / 'timing.json').read_text())
        record.update(_compare(args.output / scenario / 'baseline',
                               args.output / scenario / 'optimized'))
        before, after = (record['results'][name]['timing'] for name in roots)
        record['warm_speedup'] = before['median_ms'] / after['median_ms']
        record['first_call_speedup'] = before['first_call_ms'] / after['first_call_ms']
        record['peak_rss_ratio'] = (record['results']['optimized']['peak_process_rss_bytes']
                                    / record['results']['baseline']['peak_process_rss_bytes'])
        record['equivalent'] = True
        report['scenarios'].append(record)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        sys.stdout.write(json.dumps({key: record[key] for key in
                                     ('scenario', 'warm_speedup', 'max_absolute_error')}) + '\n')
        sys.stdout.flush()
    if digests != {name: _code_digest(root) for name, root in roots.items()}:
        raise RuntimeError('Implementation changed during measurement; campaign is invalid')
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    report['equivalent'] = True
    report['performance_passed'] = all(record['warm_speedup'] > MIN_WARM_SPEEDUP
                                       for record in report['scenarios'])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    _assert_campaign_contract(report, args.scenarios)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--optimized-root', type=Path, default=ROOT)
    parser.add_argument('--input', type=Path,
                        default=ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet')
    parser.add_argument('--target-input', type=Path,
                        default=ROOT / 'tests/fixtures/spot_15m_20250101_20250531.parquet')
    parser.add_argument('--funding-input', type=Path,
                        default=ROOT / 'tests/fixtures/btcusdt_funding_20240101_20240108.json')
    parser.add_argument('--strategy', default='ema_cross')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenarios', choices=SCENARIOS, nargs='+', default=SCENARIOS)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=32)
    parser.add_argument('--target-rows', type=int, default=128)
    args = parser.parse_args()
    if args.repeats < 3 or args.rounds < 1 or args.target_rows < 2:
        parser.error('Require repeats >= 3, rounds >= 1 and target-rows >= 2')
    for name in ('baseline_root', 'optimized_root', 'input', 'target_input', 'funding_input', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    for name in THREADS:
        os.environ[name] = '1'
    campaign(args)


if __name__ == '__main__':
    main()
