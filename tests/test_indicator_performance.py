"""Baseline contracts and execution regressions on recorded BTCUSDT bars."""
from __future__ import annotations

import hashlib
import importlib
import inspect
import os
import statistics
import subprocess
import json
import sys
import zipfile
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from limen import indicators
from tests import benchmark_indicators

FIXTURES = Path(__file__).parent / 'fixtures'
REFERENCE = json.loads((FIXTURES / 'indicator_equivalence.json').read_text())
ARCHIVE = FIXTURES / 'indicator_equivalence_reference.zip'
sys.path.insert(0, str(ARCHIVE))
try:
    BASELINES = {name: importlib.import_module(f'_limen_930_baseline.{name}')
                 for name in REFERENCE['signatures']}
finally:
    sys.path.remove(str(ARCHIVE))
SOURCE = FIXTURES.parent.parent / REFERENCE['input']
DATA = pl.read_parquet(SOURCE).select('open', 'high', 'low', 'close', 'volume')


def _case_frame(case):
    if case['variant'] == 'long':
        return pl.read_parquet(FIXTURES / 'indicator_equivalence_long.parquet')
    frame = DATA.head(case['rows'])
    variant = case['variant']
    if variant == 'flat':
        return frame.with_columns(pl.lit(DATA['close'][0]).alias(c)
                                  for c in ('open', 'high', 'low', 'close'))
    if variant == 'ties':
        return frame.with_columns(pl.col('close').alias('open'))
    if variant in ('float32', 'integer'):
        return frame.with_columns(pl.all().cast(pl.Float32 if variant == 'float32' else pl.Int64))
    if variant.startswith(('nan', 'null')):
        index = 0 if variant.endswith('head') else 70
        missing = np.nan if variant.startswith('nan') else None
        column = case.get('missing_column', 'close')
        return frame.with_columns(pl.when(pl.int_range(pl.len()) == index)
                                  .then(pl.lit(missing, dtype=pl.Float64))
                                  .otherwise(pl.col(column)).alias(column))
    return frame


@pytest.mark.parametrize('case', REFERENCE['cases'], ids=lambda c: f'{c["name"]}-{c["variant"]}-{c["rows"]}-{c["kwargs"]}')
def test_baseline_equivalence(case):
    function = getattr(indicators, case['name'])
    assert str(inspect.signature(function)) == REFERENCE['signatures'][case['name']]
    frame = _case_frame(case)
    if 'error' not in case:
        baseline = getattr(BASELINES[case['name']], case['name'])(frame, **case['kwargs'])
    for source in (frame, frame.lazy()):
        if 'error' in case:
            with pytest.raises((ValueError, ZeroDivisionError, pl.exceptions.PolarsError)) as caught:
                result = function(source, **case['kwargs'])
                if isinstance(result, pl.LazyFrame):
                    result.collect()
            assert type(caught.value).__name__ == case['error']
            assert str(caught.value).split('\n')[0] == case['error_message']
        else:
            result = function(source, **case['kwargs'])
            if isinstance(result, pl.LazyFrame):
                result = result.collect()
            assert result.columns == frame.columns + case['columns']
            assert_frame_equal(result.select(frame.columns), frame)
            assert [str(result[c].dtype) for c in case['columns']] == case['dtypes']
            for column in case['columns']:
                assert result[column].null_count() == baseline[column].null_count()
                np.testing.assert_array_equal(result[column].to_numpy(), baseline[column].to_numpy())



def test_fixture_provenance():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == REFERENCE['input_sha256']
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == REFERENCE['reference_archive_sha256']
    with zipfile.ZipFile(ARCHIVE) as saved:
        for path, digest in REFERENCE['reference_sources'].items():
            source = saved.read(f'_limen_930_baseline/{Path(path).name}')
            original = source.replace(b'from _limen_930_baseline._hilbert import',
                                      b'from limen.indicators._hilbert import')
            assert hashlib.sha256(original).hexdigest() == digest
    assert hashlib.sha256((FIXTURES / 'indicator_equivalence_long.parquet').read_bytes()).hexdigest() == REFERENCE['long_input_sha256']


@pytest.mark.parametrize(('name', 'kernel', 'columns'), [
    ('ht_sine', '_ht_sine_from_values', ('ht_sine', 'ht_sine_lead')),
    ('ht_phasor', '_ht_phasor_from_values', ('ht_phasor_inphase', 'ht_phasor_quadrature')),
    ('mama', 'mama_from_values', ('mama', 'fama')),
])
def test_paired_kernel_runs_once(monkeypatch, name, kernel, columns):
    module = importlib.import_module(f'limen.indicators.{name}')
    original = getattr(module, kernel)
    calls = []

    def calculate(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(module, kernel, calculate)
    # Pre-existing outputs must be replaced in place, even when they are interleaved.
    frame = DATA.head(256).with_columns(DATA['open'].head(256).alias(columns[0]),
                                       DATA['close'].head(256).alias(columns[1]))
    frame = frame.select(columns[1], 'close', columns[0], 'open', 'high', 'low', 'volume')
    expected = getattr(indicators, name)(DATA.head(256))
    calls.clear()
    for source in (frame, pl.concat([frame.head(100), frame.slice(100)], rechunk=False).lazy()):
        result = getattr(indicators, name)(source)
        if isinstance(result, pl.LazyFrame):
            result = result.collect()
        assert len(calls) == 1
        calls.clear()
        assert result.columns == frame.columns
        assert_frame_equal(result.select(columns), expected.select(columns))


@pytest.mark.parametrize('name', ['cci', 'linearreg', 'linearreg_angle',
                                  'linearreg_intercept', 'linearreg_slope', 'tsf'])
@pytest.mark.parametrize('period', [1, 100001])
def test_period_validation(name, period):
    with pytest.raises(ValueError, match=f'{name} period must be between 2 and 100000'):
        getattr(indicators, name)(DATA.head(40), period=period)


@pytest.mark.parametrize('parameter', ['fast_limit', 'slow_limit'])
@pytest.mark.parametrize('value', [0.0, 1.0])
def test_mama_validation(parameter, value):
    with pytest.raises(ValueError, match=f'mama {parameter} must be between 0.01 and 0.99'):
        indicators.mama(DATA.head(40), **{parameter: value})



def _timed_calculation(name):
    frame = DATA.head(3000)
    baseline = getattr(BASELINES[name], name)
    calculate = getattr(indicators, name)
    before, after = [], []
    for _ in range(3):
        started = time.perf_counter()
        expected = baseline(frame)
        before.append(time.perf_counter() - started)
        started = time.perf_counter()
        result = calculate(frame)
        after.append(time.perf_counter() - started)
    assert_frame_equal(result, expected, check_exact=True)
    return before, after


@pytest.fixture(scope='module')
def timing_worker():
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn')) as worker:
        yield worker


@pytest.mark.parametrize('name', REFERENCE['signatures'])
def test_calculation_latency(name, timing_worker):
    before, after = timing_worker.submit(_timed_calculation, name).result(timeout=60)
    threshold = 0.5 if name == 'cci' else 0.9
    assert statistics.median(after) < statistics.median(before) * threshold


def test_benchmark_relative_paths(monkeypatch, tmp_path):
    (tmp_path / 'input.parquet').symlink_to(SOURCE.resolve())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['benchmark_indicators', '--input', 'input.parquet',
                                    '--output', 'measurements', '--reference', 'HEAD'])

    def small_campaign(args, baseline):
        args.output.mkdir(parents=True)
        subprocess.run([sys.executable, str(Path(benchmark_indicators.__file__).resolve()),
                        '--worker', '--code-root', str(baseline), '--input', str(args.input),
                        '--rows', '64', '--indicator', 'cci', '--output',
                        str(args.output / 'contract.npz'), '--repeats', '1'],
                       cwd=baseline, check=True,
                       env={**os.environ, **dict.fromkeys(benchmark_indicators.THREADS, '1')})

    monkeypatch.setattr(benchmark_indicators, '_campaign', small_campaign)
    benchmark_indicators.main()
    with np.load(tmp_path / 'measurements' / 'contract.npz') as result:
        np.testing.assert_array_equal(result['cci_14'], indicators.cci(DATA.head(64))['cci_14'].to_numpy())
