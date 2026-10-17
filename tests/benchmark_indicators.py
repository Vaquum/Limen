"""Compare indicator latency and memory against a recorded commit on real bars."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ('ht_sine', 'ht_trendmode', 'ht_dcphase', 'ht_phasor',
           'ht_trendline', 'ht_dcperiod', 'mama', 'cci', 'cdladvancedblock',
           'cdl3whitesoldiers', 'cdlstalledpattern', 'linearreg',
           'linearreg_angle', 'linearreg_intercept', 'linearreg_slope', 'tsf')
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'POLARS_MAX_THREADS')


def _checkout(reference, destination):
    archive = subprocess.check_output(
        ['git', 'archive', '--format=tar', reference, 'limen'], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for member in source.getmembers():
            if member.isfile():
                target = destination / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = source.extractfile(member)
                if stream is None:
                    raise ValueError(f'Unreadable archive member: {member.name}')
                target.write_bytes(stream.read())


def _worker(args):
    sys.path.insert(0, str(args.code_root))
    started = time.perf_counter()
    import numpy as np
    import polars as pl
    dependency_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    import limen.indicators as indicators
    import_ms = (time.perf_counter() - started) * 1000
    data = pl.read_parquet(args.input).head(args.rows)
    if data.height != args.rows:
        raise ValueError(f'Input has fewer than {args.rows} real rows')
    digest = hashlib.sha256(data.select('open', 'high', 'low', 'close', 'volume')
                            .to_numpy().tobytes()).hexdigest()
    calculate = getattr(indicators, args.indicator)
    started = time.perf_counter()
    result = calculate(data)
    first_ms = (time.perf_counter() - started) * 1000
    times = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        calculate(data)
        times.append((time.perf_counter() - started) * 1000)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mib = rss / (1024 ** 2 if sys.platform == 'darwin' else 1024)
    tracemalloc.start()
    calculate(data)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    np.savez(args.output, **{name: result[name].to_numpy()
                            for name in result.columns if name not in data.columns})
    return {'indicator': args.indicator, 'rows': args.rows, 'input_sha256': digest,
            'dependency_import_ms': dependency_ms, 'indicator_import_ms': import_ms,
            'first_call_ms': first_ms, 'steady_ms': statistics.median(times),
            'samples_ms': times, 'peak_rss_mib': rss_mib,
            'traced_peak_mib': peak_bytes / 1024 ** 2,
            'schema': {name: str(dtype) for name, dtype in result.schema.items()},
            'polars_threads': pl.thread_pool_size(),
            'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                         'polars': pl.__version__}}


def _campaign(args, baseline):
    import numpy as np

    args.output.mkdir(parents=True, exist_ok=True)
    report = {'reference': args.reference, 'candidate': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'input': str(args.input.resolve()),
        'file_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'hardware': platform.platform(), 'machine': platform.machine(),
        'processor': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                            text=True).strip() if sys.platform == 'darwin' else platform.processor(),
        'threads': dict.fromkeys(THREADS, '1'), 'results': []}
    for rows in (20000, 83136):
        for name in args.indicators:
            pair = {}
            for label, code_root in (('before', baseline), ('after', ROOT)):
                output = args.output / f'{rows}-{name}-{label}.npz'
                command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                           '--code-root', str(code_root), '--input', str(args.input),
                           '--rows', str(rows), '--indicator', name, '--output', str(output),
                           '--repeats', str(args.repeats)]
                completed = subprocess.run(command, cwd=code_root, check=True,
                                           text=True, capture_output=True,
                                           env={**os.environ, **dict.fromkeys(THREADS, '1')})
                if completed.stderr:
                    sys.stderr.write(completed.stderr)
                pair[label] = json.loads(completed.stdout)
            before, after = pair['before'], pair['after']
            assert before['input_sha256'] == after['input_sha256']
            assert before['schema'] == after['schema']
            with (np.load(args.output / f'{rows}-{name}-before.npz') as left,
                  np.load(args.output / f'{rows}-{name}-after.npz') as right):
                assert left.files == right.files
                for column in left.files:
                    np.testing.assert_array_equal(left[column], right[column],
                                                  err_msg=f'{name}: {column}, {rows} rows')
            report['results'].append(pair)
            (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            sys.stdout.write(f'{rows} {name}: {before["steady_ms"]:.2f} -> '
                             f'{after["steady_ms"]:.2f} ms; exact outputs\n')
            sys.stdout.flush()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--reference', default='fe27656c')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--indicators', nargs='+', default=TARGETS)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--code-root', type=Path, default=ROOT, help=argparse.SUPPRESS)
    parser.add_argument('--rows', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--indicator', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        sys.stdout.write(json.dumps(_worker(args)) + '\n')
    else:
        with tempfile.TemporaryDirectory(prefix='limen-indicators-') as directory:
            baseline = Path(directory)
            _checkout(args.reference, baseline)
            _campaign(args, baseline)


if __name__ == '__main__':
    main()
