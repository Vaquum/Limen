"""Candlestick equivalence and scalar-dispatch costs on recorded market bars."""
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
ARCHIVE = FIXTURES / 'candles_performance_reference.zip'
REFERENCE = json.loads((FIXTURES / 'candles_performance_reference.json').read_text())
NAMES = [Path(p).stem for p in REFERENCE['sources']]
sys.path.insert(0, str(ARCHIVE))
try:
    BASELINES = {name: getattr(importlib.import_module(f'_limen_candles_baseline.{name}'), name)
                 for name in NAMES}
finally:
    sys.path.remove(str(ARCHIVE))
DATA = pl.read_parquet(FIXTURES / 'spot_15m_20250101_20250531.parquet').select('open', 'high', 'low', 'close')


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('variant', ['raw', 'empty', 'short', 'flat', 'ties', 'nan', 'null', 'integer', 'float32', 'columns'])
def test_baseline_equivalence(name, variant):
    from limen import indicators
    calculate = getattr(indicators, name)
    assert inspect.signature(calculate) == inspect.signature(BASELINES[name])
    frame = DATA.head(512)
    kwargs = {}
    if variant in ('empty', 'short'):
        frame = frame.head(0 if variant == 'empty' else 11)
    elif variant == 'flat':
        frame = frame.with_columns(pl.lit(DATA['close'][0]).alias(c) for c in frame.columns)
    elif variant == 'ties':
        frame = frame.with_columns(pl.col('close').alias('open'))
    elif variant in ('nan', 'null'):
        frame = frame.with_columns(pl.when(pl.int_range(pl.len()).is_in([0, 70]))
                                   .then(pl.lit(np.nan if variant == 'nan' else None, dtype=pl.Float64))
                                   .otherwise(pl.col(c)).alias(c) for c in frame.columns)
    elif variant in ('integer', 'float32'):
        frame = frame.with_columns(pl.all().cast(pl.Int64 if variant == 'integer' else pl.Float32))
    elif variant == 'columns':
        frame = frame.rename({c: 'input_' + c for c in frame.columns})
        kwargs = {c + '_col': 'input_' + c for c in ('open', 'high', 'low', 'close')}
    expected = BASELINES[name](frame, **kwargs)
    chunked = pl.concat([frame.head(100), frame.slice(100)], rechunk=False)
    for source in (chunked, chunked.lazy()):
        result = calculate(source, **kwargs)
        if isinstance(result, pl.LazyFrame):
            result = result.collect()
        assert_frame_equal(result, expected, check_exact=True)


def test_reference_provenance():
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == REFERENCE['archive_sha256']
    with zipfile.ZipFile(ARCHIVE) as saved:
        for path, digest in REFERENCE['sources'].items():
            assert hashlib.sha256(saved.read('_limen_candles_baseline/' + Path(path).name)).hexdigest() == digest


def _timings(name):
    from limen import indicators
    frame = DATA.head(3000)
    before, after = [], []
    for _ in range(3):
        started = time.process_time()
        expected = BASELINES[name](frame)
        before.append(time.process_time() - started)
        started = time.process_time()
        result = getattr(indicators, name)(frame)
        after.append(time.process_time() - started)
    assert_frame_equal(result, expected, check_exact=True)
    return before, after


@pytest.fixture(scope='module')
def timing_worker():
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn')) as worker:
        yield worker


@pytest.mark.parametrize('name', NAMES)
def test_calculation_latency(name, timing_worker):
    before, after = timing_worker.submit(_timings, name).result(timeout=60)
    assert statistics.median(after) < statistics.median(before) * .9
