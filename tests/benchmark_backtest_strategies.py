"""Benchmark common long/flat strategies only after full execution parity."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from tests.backtest_benchmark import THREADS

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ('spot_1h_20240101_20241231.parquet', 'spot_15m_20250101_20250531.parquet')
STRATEGIES = ('buy_and_hold', 'sma_cross', 'ema_cross', 'macd', 'rsi_mean_reversion',
              'bollinger_mean_reversion', 'donchian_breakout', 'time_series_momentum')
ENGINES = ('limen', 'vectorbt', 'vectorbt_rust', 'backtesting_py', 'backtrader')


def _git(*arguments):
    return subprocess.check_output(['git', *arguments], cwd=ROOT, text=True).strip()


def _code_digest():
    digest = hashlib.sha256()
    paths = [*ROOT.glob('limen/**/*.py'), *ROOT.glob('tests/backtest_benchmark/*.py'),
             Path(__file__).resolve()]
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _engine_pythons(values):
    result = dict.fromkeys(ENGINES, sys.executable)
    for value in values:
        engine, separator, executable = value.partition('=')
        if not separator or engine not in ENGINES or not Path(executable).is_file():
            raise ValueError('--engine-python requires ENGINE=/existing/path/to/python')
        # Resolving a venv's python symlink would discard that environment.
        result[engine] = str(Path(executable).absolute())
    return result


def _save_report(output, report):
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    columns = ('input', 'strategy', 'evaluation_rows', 'engine', 'status',
               'cold_engine_ms', 'cold_common_outputs_ms', 'warm_engine_median_ms',
               'warm_engine_p95_ms', 'warm_common_median_ms', 'warm_common_p95_ms',
               'snapshot_median_ms', 'peak_rss_mib', 'total_return', 'max_drawdown',
               'entries', 'max_abs_equity_error', 'reason')
    with (output / 'report.csv').open('w', newline='') as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        for scenario in report['scenarios']:
            for result in scenario['results']:
                row = {name: result.get(name) for name in
                       ('engine', 'status', 'peak_rss_mib', 'max_abs_equity_error', 'reason')}
                row.update({name: scenario[name] for name in ('input', 'strategy', 'evaluation_rows')})
                row.update({name: result.get('metrics', {}).get(name) for name in
                            ('total_return', 'max_drawdown', 'entries')})
                row['cold_engine_ms'] = result.get('cold', {}).get('engine_ms')
                row['cold_common_outputs_ms'] = result.get('cold', {}).get('common_outputs_ms')
                for key, scope in (('engine', 'engine_ms'), ('common', 'common_outputs_ms')):
                    for statistic in ('median', 'p95'):
                        row[f'warm_{key}_{statistic}_ms'] = result.get('warm', {}).get(scope, {}).get(f'{statistic}_ms')
                row['snapshot_median_ms'] = result.get('native_snapshot', {}).get('warm', {}).get('median_ms')
                writer.writerow(row)


def campaign(args):
    from limen import __version__
    from limen.sfd.rule_based.common_strategies import STRATEGY_PARAMETERS
    from tests.backtest_benchmark.contract import (
        EQUITY_ATOL, EQUITY_RTOL, INITIAL_CASH, assert_parity, load_result,
    )
    from tests.backtest_benchmark.prepare import prepare_scenario, read_bars
    from tests.backtest_benchmark.worker import package_versions

    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output directory must be new or empty; prior measurements are immutable')
    args.output.mkdir(parents=True, exist_ok=True)
    executables = _engine_pythons(args.engine_python)
    report = {
        'schema_version': 1, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'commit': _git('rev-parse', 'HEAD'), 'working_tree': _git('status', '--porcelain'),
        'code_sha256': _code_digest(), 'limen_version': __version__,
        'platform': platform.platform(), 'machine': platform.machine(),
        'processor': platform.processor(), 'logical_cpus': os.cpu_count(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'versions': package_versions(), 'engine_pythons': executables,
        'repeats': args.repeats, 'threads': dict.fromkeys(THREADS, '1'),
        'strategy_parameters': {key: dict(value) for key, value in STRATEGY_PARAMETERS.items()},
        'excluded_inputs': [{
            'input': 'tests/fixtures/indicator_equivalence_long.parquet',
            'reason': 'OHLCV-only indicator fixture has no recorded timestamps; chronological and event-clock validation is unavailable',
        }],
        'contract': {
            'initial_cash': INITIAL_CASH, 'fee_bps': 0, 'slip_bps': 0,
            'liquidity': 'unconstrained fills at the recorded close; volume participation is not modeled',
            'execution_lag_bars': 1, 'position': 'all-in long or flat',
            'timing': 'signal at close t holds close t to close t+1; final evaluated close liquidates',
            'settlement': 'one additional real source bar; excluded from evaluation and strategy preparation',
            'equity_rtol': EQUITY_RTOL, 'equity_atol': EQUITY_ATOL,
            'parity': 'every equity sample, held bar, fill bar, side and price; independent cash replay and Limen',
            'costed_comparison': 'unsupported: legacy snapshot entry-fee accounting differs from affordable cash-spot sizing',
            'timing_scopes': 'engine API; API plus common equity/fills/summary extraction; Limen full snapshot separately',
            'cold': 'fresh process and unique empty Numba disk cache; imports/preparation reported separately',
            'memory': 'process high-water RSS, including imports and preparation; not allocation-only engine memory',
            'comparison_limit': 'public APIs perform unequal additional native work; no universal engine ranking',
            'tail_statistic_limit': 'p95 is a sample percentile; few repeats do not establish tail latency',
        }, 'scenarios': [],
    }
    failed = False
    for source in args.inputs:
        started = time.perf_counter_ns()
        data = read_bars(source, args.rows)
        load_ms = (time.perf_counter_ns() - started) / 1e6
        file_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        for strategy in args.strategies:
            name = f'{source.stem}-{file_digest[:8]}-{data.height - 1}-{strategy}'
            folder = args.output / name
            folder.mkdir()
            prepared = folder / 'prepared.npz'
            scenario = prepare_scenario(data, strategy, prepared, args.repeats)
            scenario.update(input=str(source), input_file_sha256=file_digest,
                            parquet_read_validate_ms=load_ms, results=[])
            report['scenarios'].append(scenario)
            baseline = None
            for engine in args.engines:
                worker_output = folder / f'{engine}.json'
                cache = folder / f'{engine}-numba-cache'
                cache.mkdir()
                command = [executables[engine], '-m', 'tests.benchmark_backtest_strategies',
                           '--worker', '--engine', engine, '--prepared', str(prepared),
                           '--worker-output', str(worker_output), '--repeats', str(args.repeats)]
                started = time.perf_counter_ns()
                process = subprocess.run(command, cwd=ROOT, check=False,
                                         env={**os.environ, **dict.fromkeys(THREADS, '1'),
                                              'PYTHONHASHSEED': '0', 'NUMBA_CACHE_DIR': str(cache)})
                if worker_output.exists():
                    result = json.loads(worker_output.read_text())
                else:
                    result = {'engine': engine, 'status': 'error',
                              'reason': f'Worker exited {process.returncode} without a report; see stderr'}
                result['process_wall_ms'] = (time.perf_counter_ns() - started) / 1e6
                result['worker_returncode'] = process.returncode
                if result['status'] == 'verified':
                    actual = load_result(worker_output.with_suffix('.npz'))
                    if engine == 'limen':
                        baseline = actual
                    elif baseline is None:
                        result.update(status='unranked', reason='Limen baseline did not verify')
                    else:
                        try:
                            assert_parity(actual, baseline)
                        except AssertionError as exc:
                            result.update(status='parity_failed', reason=str(exc))
                failed = failed or result['status'] in ('parity_failed', 'error', 'unranked')
                if engine == 'limen' and result['status'] != 'verified':
                    failed = True
                scenario['results'].append(result)
                _save_report(args.output, report)
                sys.stdout.write(f'{source.name} {strategy} {engine}: {result["status"]}\n')
                sys.stdout.flush()
    if report['code_sha256'] != _code_digest():
        raise RuntimeError('Code changed during measurement; this campaign is invalid')
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    unavailable = any(result['status'] == 'unavailable' for scenario in report['scenarios']
                      for result in scenario['results'])
    report['status'] = 'failed' if failed else (
        'completed_with_unavailable_engines' if unavailable else 'completed')
    _save_report(args.output, report)
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+',
                        default=[ROOT / 'tests/fixtures' / name for name in FIXTURES])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--rows', type=int, help='Evaluation rows; requires one extra real settlement row')
    parser.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=STRATEGIES)
    parser.add_argument('--engines', nargs='+', choices=ENGINES,
                        default=('limen', 'vectorbt', 'backtesting_py', 'backtrader'))
    parser.add_argument('--engine-python', action='append', default=[], metavar='ENGINE=PYTHON')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--engine', choices=ENGINES, help=argparse.SUPPRESS)
    parser.add_argument('--prepared', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for variable in THREADS:
        os.environ[variable] = '1'
    if args.repeats < 3:
        parser.error('--repeats must be at least 3; p95 from few samples is descriptive only')
    if args.worker:
        from tests.backtest_benchmark.worker import run_worker
        return run_worker(args)
    if args.output is None:
        parser.error('--output is required')
    if args.rows is not None and args.rows < 2:
        parser.error('--rows must be at least 2')
    args.output = args.output.resolve()
    args.inputs = [path.resolve() for path in args.inputs]
    args.engines = ['limen', *(engine for engine in args.engines if engine != 'limen')]
    return campaign(args)


if __name__ == '__main__':
    raise SystemExit(main())
