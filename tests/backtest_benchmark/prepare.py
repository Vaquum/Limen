"""Recorded-data provenance and common causal strategy preparation."""
from __future__ import annotations

import hashlib
import time

import numpy as np
import polars as pl

from limen.sfd.rule_based.common_strategies import strategy_signals
from tests.backtest_benchmark.contract import targets
from tests.backtest_benchmark.worker import timing_summary


def read_bars(path, rows=None):
    data = pl.read_parquet(path, n_rows=None if rows is None else rows + 1)
    if rows is not None and data.height != rows + 1:
        raise ValueError(f'{path} needs {rows} evaluation bars plus one real settlement bar')
    if data.height < 3:
        raise ValueError('At least two evaluation bars plus one real settlement bar are required')
    required = ('datetime', 'open', 'high', 'low', 'close', 'volume')
    if set(required) - set(data.columns):
        raise ValueError(f'Recorded bars require columns {required}')
    if data.select(required).null_count().sum_horizontal()[0]:
        raise ValueError('Recorded bars contain null timestamps or OHLCV')
    if data['datetime'].dtype.base_type() != pl.Datetime:
        raise ValueError('datetime must have an explicit Polars Datetime unit')
    timestamps = data['datetime'].dt.epoch('ns').to_numpy()
    if not np.all(timestamps[1:] > timestamps[:-1]):
        raise ValueError('Recorded timestamps must be strictly increasing, without duplicates')
    prices = data.select('open', 'high', 'low', 'close').to_numpy()
    if not np.isfinite(prices).all() or not (prices > 0).all():
        raise ValueError('Recorded OHLC must be finite and positive')
    volume = data['volume'].to_numpy()
    if not np.isfinite(volume).all() or (volume < 0).any():
        raise ValueError('Recorded volume must be finite and nonnegative')
    if ((prices[:, 1] < np.max(prices[:, (0, 2, 3)], axis=1)).any()
            or (prices[:, 2] > np.min(prices[:, (0, 1, 3)], axis=1)).any()):
        raise ValueError('Recorded OHLC range is inconsistent')
    return data


def prepare_scenario(data, strategy, path, repeats):
    evaluation = data.head(data.height - 1)
    started = time.perf_counter_ns()
    signal = strategy_signals(evaluation, strategy)
    first_ms = (time.perf_counter_ns() - started) / 1e6
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        repeated = strategy_signals(evaluation, strategy)
        samples.append((time.perf_counter_ns() - started) / 1e6)
        np.testing.assert_array_equal(repeated, signal)
    columns = {name: data[name].to_numpy().astype(float) for name in
               ('open', 'high', 'low', 'close', 'volume')}
    columns['timestamp_ns'] = data['datetime'].dt.epoch('ns').to_numpy()
    columns['signal'] = targets(np.r_[signal, np.int8(0)])
    np.savez(path, **columns)
    digest = hashlib.sha256()
    for name, values in sorted(columns.items()):
        digest.update(name.encode())
        digest.update(str(values.dtype).encode())
        digest.update(values.tobytes())
    return {'strategy': strategy, 'evaluation_rows': evaluation.height,
            'source_rows_including_settlement': data.height,
            'evaluation_start': str(evaluation['datetime'][0]),
            'evaluation_end': str(evaluation['datetime'][-1]),
            'settlement_timestamp': str(data['datetime'][-1]),
            'prepared_arrays_sha256': digest.hexdigest(),
            'signals_sha256': hashlib.sha256(signal.tobytes()).hexdigest(),
            'signal_preparation': {'first_call_ms': first_ms,
                                   'warm': timing_summary(samples)},
            'prepared_path': str(path)}
