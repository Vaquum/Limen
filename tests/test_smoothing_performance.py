"""Exact smoothing contracts on recorded BTCUSDT bars."""
import hashlib
import importlib
import inspect
import json
import statistics
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

FIXTURES = Path(__file__).parent / 'fixtures'
ARCHIVE = FIXTURES / 'smoothing_performance_reference.zip'
sys.path.insert(0, str(ARCHIVE))
try:
    BASELINES = importlib.import_module('_limen_smoothing_baseline')
finally:
    sys.path.remove(str(ARCHIVE))
DATA = pl.read_parquet(FIXTURES / 'spot_15m_20250101_20250531.parquet').select('open', 'high', 'low', 'close')
CASES = [('sma', {'period': p}) for p in (2, 30, 168, 510, 100000)]
CASES += [(name, {'ma_type': kind}) for name in ('apo', 'ppo') for kind in range(9)]
CASES += [('macdext', kwargs) for kwargs in ({}, {'fast_ma_type': 7, 'slow_ma_type': 7},
                                          {'fast_ma_type': 7, 'slow_ma_type': 0},
                                          {'fast_period': 26, 'slow_period': 12,
                                           'fast_ma_type': 7, 'slow_ma_type': 7, 'signal_ma_type': 7})]



@pytest.mark.parametrize(('name', 'kwargs'), CASES)
@pytest.mark.parametrize('variant', ['raw', 'empty', 'short', 'flat', 'ties', 'nan_head', 'nan_mid', 'null', 'float32'])
def test_baseline_equivalence(name, kwargs, variant):
    from limen import indicators
    calculate, baseline = getattr(indicators, name), getattr(BASELINES, name)
    assert inspect.signature(calculate) == inspect.signature(baseline)
    frame = DATA.head(512)
    if variant in ('empty', 'short'):
        frame = frame.head(0 if variant == 'empty' else 3)
    elif variant == 'flat':
        frame = frame.with_columns(pl.lit(DATA['close'][0]).alias(c) for c in frame.columns)
    elif variant == 'ties':
        frame = frame.with_columns(pl.col('close').alias('high'), pl.col('close').alias('low'))
    elif variant in ('nan_head', 'nan_mid', 'null'):
        index = 0 if variant == 'nan_head' else 70
        frame = frame.with_columns(pl.when(pl.int_range(pl.len()) == index)
                                   .then(pl.lit(None if variant == 'null' else np.nan, dtype=pl.Float64))
                                   .otherwise(pl.col(c)).alias(c) for c in frame.columns)
    elif variant == 'float32':
        frame = frame.with_columns(pl.all().cast(pl.Float32))
    expected = baseline(frame, **kwargs)
    for source in (frame, frame.lazy()):
        result = calculate(source, **kwargs)
        if isinstance(result, pl.LazyFrame):
            result = result.collect()
        assert_frame_equal(result, expected, check_exact=True)


def test_reference_provenance():
    metadata = json.loads((FIXTURES / 'smoothing_performance_reference.json').read_text())
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == metadata['archive_sha256']
    with zipfile.ZipFile(ARCHIVE) as saved:
        for path, digest in metadata['sources'].items():
            source = saved.read('_limen_smoothing_baseline/' + Path(path).name).replace(
                b'from _limen_smoothing_baseline.', b'from limen.indicators.')
            assert hashlib.sha256(source).hexdigest() == digest


def _timings(name):
    from limen import indicators
    frame = DATA.head(3000)
    kwargs = {'period': 168} if name == 'sma' else ({'ma_type': 7} if name in ('apo', 'ppo') else {'fast_ma_type': 7, 'slow_ma_type': 7})
    before, after = [], []
    for _ in range(3):
        started = time.process_time()
        expected = getattr(BASELINES, name)(frame, **kwargs)
        before.append(time.process_time() - started)
        started = time.process_time()
        result = getattr(indicators, name)(frame, **kwargs)
        after.append(time.process_time() - started)
    assert_frame_equal(result, expected, check_exact=True)
    return before, after


@pytest.fixture(scope='module')
def timing_worker():
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn')) as worker:
        yield worker


@pytest.mark.parametrize('name', ['sma', 'apo', 'ppo', 'macdext'])
def test_calculation_latency(name, timing_worker):
    before, after = timing_worker.submit(_timings, name).result(timeout=60)
    assert statistics.median(after) < statistics.median(before) * .9


@pytest.mark.parametrize('name', ['apo', 'ppo'])
def test_input_column_collision(name):
    from limen import indicators
    frame = DATA.head(128).rename({'close': 'ma_12_7'})
    kwargs = {'price_col': 'ma_12_7', 'ma_type': 7}
    expected = getattr(BASELINES, name)(frame, **kwargs)
    assert_frame_equal(getattr(indicators, name)(frame, **kwargs), expected, check_exact=True)


@pytest.mark.parametrize(('name', 'kwargs', 'count'), [
    ('apo', {'ma_type': 7}, 1), ('ppo', {'ma_type': 7}, 1),
    ('macdext', {'fast_ma_type': 7, 'slow_ma_type': 7}, 1),
])
def test_mama_runs_once(monkeypatch, name, kwargs, count):
    from limen import indicators
    module = importlib.import_module('limen.indicators.ma')
    original = module.mama_from_values
    calls = []

    def calculate(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(module, 'mama_from_values', calculate)
    for source in (DATA.head(128), DATA.head(128).lazy()):
        calls.clear()
        result = getattr(indicators, name)(source, **kwargs)
        if isinstance(result, pl.LazyFrame):
            result.collect()
        assert len(calls) == count


def test_longest_period():
    from limen.indicators.sma import sma
    frame = pl.read_parquet(FIXTURES / 'indicator_equivalence_long.parquet')
    assert_frame_equal(sma(frame, period=100000), BASELINES.sma(frame, period=100000), check_exact=True)
