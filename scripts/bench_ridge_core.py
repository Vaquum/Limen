from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

from limen.data import HistoricalData


ROOT = Path(__file__).resolve().parents[1]
RIDGE_MANIFEST = ROOT / 'rust' / 'ridge_core' / 'Cargo.toml'
RIDGE_BIN = ROOT / 'rust' / 'ridge_core' / 'target' / 'release' / 'ridge_core'
MAGIC = b'RIDGE001'
MIN_PREPARED_ROWS = 1_000
SKLEARN_COEFF_TOLERANCE = 1e-7

FEATURE_COLUMNS = [
    'ret_1',
    'ret_4',
    'ret_16',
    'range_pct',
    'body_pct',
    'upper_wick_pct',
    'lower_wick_pct',
    'volume_log_delta',
    'trades_log_delta',
    'close_vs_sma_32',
    'close_vs_sma_96',
    'volatility_32',
]


def main() -> None:
    args = parse_args()
    if not args.no_build:
        build_rust()

    frame = load_klines(args)
    prepared = prepare_frame(frame)
    x, y = matrix_from_frame(prepared)

    with tempfile.NamedTemporaryFile(suffix='.ridge.bin') as tmp:
        write_ridge_binary(Path(tmp.name), x, y)
        verify_against_sklearn(Path(tmp.name), x, y, parse_float_list(args.verify_alphas))
        run_benchmark(Path(tmp.name), args.counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark the Rust ridge core on Limen 15m klines.')
    parser.add_argument('--start', default='2025-01-01')
    parser.add_argument('--end', default='2025-02-01')
    parser.add_argument('--counts', default='10,100,1000,10000,100000')
    parser.add_argument('--verify-alphas', default='0.000001,0.001,1,100')
    parser.add_argument('--no-build', action='store_true')
    return parser.parse_args()


def build_rust() -> None:
    cargo = shutil.which('cargo')
    if cargo is None:
        raise RuntimeError('cargo is not available on PATH')

    subprocess.run(  # noqa: S603
        [cargo, 'build', '--release', '--manifest-path', str(RIDGE_MANIFEST)],
        check=True,
        cwd=ROOT,
    )


def load_klines(args: argparse.Namespace) -> pl.DataFrame:
    data = HistoricalData().get_spot_klines(
        kline_size=900,
        start_date_limit=args.start,
        end_date_limit=args.end,
    )
    write_event('data', {
        'source': 'HistoricalData.get_spot_klines',
        'kline_size_seconds': 900,
        'start': args.start,
        'end': args.end,
        'rows': data.height,
        'cols': data.width,
        'estimated_size_bytes': data.estimated_size(),
    })
    return data


def prepare_frame(data: pl.DataFrame) -> pl.DataFrame:
    base = data.with_columns([
        (pl.col('close') / pl.col('close').shift(1) - 1.0).alias('ret_1'),
        (pl.col('close') / pl.col('close').shift(4) - 1.0).alias('ret_4'),
        (pl.col('close') / pl.col('close').shift(16) - 1.0).alias('ret_16'),
        (pl.col('high') / pl.col('low') - 1.0).alias('range_pct'),
        ((pl.col('close') - pl.col('open')) / pl.col('open')).alias('body_pct'),
        ((pl.col('high') - pl.max_horizontal('open', 'close')) / pl.col('open')).alias('upper_wick_pct'),
        ((pl.min_horizontal('open', 'close') - pl.col('low')) / pl.col('open')).alias('lower_wick_pct'),
        (pl.col('volume').log1p() - pl.col('volume').shift(1).log1p()).alias('volume_log_delta'),
        (pl.col('no_of_trades').log1p() - pl.col('no_of_trades').shift(1).log1p()).alias('trades_log_delta'),
        (pl.col('close') / pl.col('close').rolling_mean(window_size=32) - 1.0).alias('close_vs_sma_32'),
        (pl.col('close') / pl.col('close').rolling_mean(window_size=96) - 1.0).alias('close_vs_sma_96'),
        pl.col('close').pct_change().rolling_std(window_size=32).alias('volatility_32'),
    ])
    prepared = base.with_columns(
        (
            ((pl.col('close').shift(-1) / pl.col('close') - 1.0) - 0.0002)
            / pl.col('ret_1').rolling_std(window_size=96)
        ).alias('target_edge')
    ).select([*FEATURE_COLUMNS, 'target_edge']).drop_nulls()

    if prepared.height < MIN_PREPARED_ROWS:
        raise ValueError(f'prepared frame too small: {prepared.height} rows')
    target_std = prepared['target_edge'].std()
    if target_std is None or target_std <= 0:
        raise ValueError('target_edge has no variance')

    write_event('prepared', {
        'rows': prepared.height,
        'features': len(FEATURE_COLUMNS),
        'estimated_size_bytes': prepared.estimated_size(),
        'target_mean': prepared['target_edge'].mean(),
        'target_std': target_std,
    })
    return prepared


def matrix_from_frame(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = frame.select(FEATURE_COLUMNS).to_numpy().astype(np.float64, copy=True)
    y = frame['target_edge'].to_numpy().astype(np.float64, copy=True)

    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds[stds == 0.0] = 1.0
    x = (x - means) / stds

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('prepared matrix contains non-finite values')

    write_event('matrix', {
        'rows': int(x.shape[0]),
        'cols': int(x.shape[1]),
        'x_bytes': int(x.nbytes),
        'y_bytes': int(y.nbytes),
        'x_mean_abs_max': float(np.abs(x.mean(axis=0)).max()),
        'x_std_min': float(x.std(axis=0).min()),
        'x_std_max': float(x.std(axis=0).max()),
    })
    return x, y


def write_ridge_binary(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    rows, cols = x.shape
    with path.open('wb') as handle:
        handle.write(MAGIC)
        handle.write(struct.pack('<QQ', rows, cols))
        handle.write(np.ascontiguousarray(x).astype('<f8', copy=False).tobytes())
        handle.write(np.ascontiguousarray(y).astype('<f8', copy=False).tobytes())


def verify_against_sklearn(path: Path, x: np.ndarray, y: np.ndarray, alphas: list[float]) -> None:
    if not alphas:
        raise ValueError('at least one verify alpha is required')

    max_seen_diff = 0.0
    last_inspect: dict[str, Any] | None = None
    for alpha in alphas:
        rust = run_rust(['inspect', str(path), str(alpha)])
        inspect = next(item for item in rust if item['event'] == 'inspect')
        beta = np.array(inspect['beta'], dtype=np.float64)

        model = Ridge(alpha=alpha, fit_intercept=True, solver='cholesky')
        model.fit(x, y)
        expected = np.concatenate([[model.intercept_], model.coef_])
        max_abs_diff = float(np.max(np.abs(beta - expected)))
        max_seen_diff = max(max_seen_diff, max_abs_diff)
        if max_abs_diff > SKLEARN_COEFF_TOLERANCE:
            raise ValueError(f'Rust/sklearn coefficient mismatch: {max_abs_diff} at alpha={alpha}')

        if inspect['coeff_norm'] <= 0 or not np.isfinite(inspect['r2']):
            raise ValueError(f'Rust model failed sanity check: {inspect}')
        last_inspect = inspect

    if last_inspect is None:
        raise ValueError('sklearn verification produced no inspections')

    write_event('verify', {
        'alphas': alphas,
        'max_abs_coeff_diff_vs_sklearn': max_seen_diff,
        'last_rust_r2': last_inspect['r2'],
        'last_rust_mse': last_inspect['mse'],
        'last_coeff_norm': last_inspect['coeff_norm'],
    })


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(',') if item.strip()]


def run_benchmark(path: Path, counts: str) -> None:
    events = run_rust(['bench', str(path), counts])
    for event in events:
        write_event(event['event'], event)


def run_rust(args: list[str]) -> list[dict[str, Any]]:
    completed = subprocess.run(  # noqa: S603
        [str(RIDGE_BIN), *args],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]


def write_event(event: str, payload: dict[str, Any]) -> None:
    merged = {'event': event, **payload}
    sys.stdout.write(json.dumps(merged, sort_keys=True) + '\n')
    sys.stdout.flush()


if __name__ == '__main__':
    main()
