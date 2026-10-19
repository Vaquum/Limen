"""Exact line-feature contracts on recorded BTCUSDT bars."""
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
ARCHIVE = FIXTURES / 'line_performance_reference.zip'
sys.path.insert(0, str(ARCHIVE))
try:
    BASELINES = {name: importlib.import_module(f'_limen_lines_baseline.{name}')
                 for name in ('price_lines', 'quantile_price_lines', 'find_price_lines')}
finally:
    sys.path.remove(str(ARCHIVE))
DATA = pl.read_parquet(FIXTURES / 'spot_15m_20250101_20250531.parquet').select('close')


@pytest.mark.parametrize('name', ['price_lines', 'quantile_price_lines'])
@pytest.mark.parametrize(('rows', 'duration', 'height', 'variant'), [
    (0, 48, .002, 'raw'), (1, 48, .002, 'raw'), (2, 2, .002, 'raw'),
    (17, 48, .002, 'raw'), (512, 13, .005, 'raw'), (512, 48, .002, 'raw'),
    (512, 48, 1.0, 'raw'), (512, 48, .002, 'flat'),
    (512, 48, .002, 'nan'), (512, 48, .002, 'null'),
])
@pytest.mark.parametrize('research', [True, False])
def test_line_equivalence(name, rows, duration, height, variant, research):
    calculate = getattr(importlib.import_module(f'limen.features.{name}'), name)
    baseline = getattr(BASELINES[name], name)
    assert inspect.signature(calculate) == inspect.signature(baseline)
    frame = DATA.head(rows)
    if variant == 'flat':
        frame = frame.with_columns(pl.lit(DATA['close'][0]).alias('close'))
    elif variant in ('nan', 'null'):
        frame = frame.with_columns(pl.when(pl.int_range(pl.len()) == 70)
                                   .then(pl.lit(np.nan if variant == 'nan' else None, dtype=pl.Float64))
                                   .otherwise(pl.col('close')).alias('close'))
    kwargs = {'max_duration_hours': duration, 'min_height_pct': height, 'include_research_only': research}
    expected = baseline(frame, **kwargs)
    chunked = pl.concat([frame.head(rows // 2), frame.slice(rows // 2)], rechunk=False)
    for source in (chunked, chunked.lazy()):
        result = calculate(source, **kwargs)
        if isinstance(result, pl.LazyFrame):
            result = result.collect()
        if name == 'price_lines':
            assert_frame_equal(result, expected, check_exact=True)
        else:
            assert result.schema == expected.schema
            assert_frame_equal(result.tail(1), expected.tail(1), rel_tol=1e-12, abs_tol=1e-12)
            if research:
                assert result['active_quantile_count'].equals(expected['active_quantile_count'])
            assert_frame_equal(result, calculate(frame, **kwargs), check_exact=True)


@pytest.mark.parametrize('quantile', [0.0, .7, .8, 1.0])
def test_quantile_boundaries(quantile):
    from limen.features.quantile_price_lines import quantile_price_lines
    kwargs = {'max_duration_hours': 48, 'min_height_pct': .002, 'quantile_threshold': quantile,
              'density_lookback_hours': 3, 'momentum_lookback_hours': 1, 'height_lookback_hours': 2}
    expected = BASELINES['quantile_price_lines'].quantile_price_lines(DATA.head(512), **kwargs)
    result = quantile_price_lines(DATA.head(512), **kwargs)
    if quantile == 0:
        assert_frame_equal(result, expected, check_exact=True)
    else:
        assert_frame_equal(result.tail(1), expected.tail(1), rel_tol=1e-12, abs_tol=1e-12)
        assert result['active_quantile_count'].equals(expected['active_quantile_count'])


def test_dictionary_contract_and_provenance():
    from limen.utils.find_price_lines import find_price_lines
    close = DATA['close'].head(512).to_numpy()
    assert find_price_lines(close, 48, .002) == BASELINES['find_price_lines'].find_price_lines(close, 48, .002)
    metadata = json.loads((FIXTURES / 'line_performance_reference.json').read_text())
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == metadata['archive_sha256']
    with zipfile.ZipFile(ARCHIVE) as saved:
        for path, digest in metadata['sources'].items():
            source = saved.read(f'_limen_lines_baseline/{Path(path).name}').replace(
                b'from _limen_lines_baseline.find_price_lines import',
                b'from limen.utils.find_price_lines import').replace(
                b'from _limen_lines_baseline.filter_lines_by_quantile import',
                b'from limen.utils.filter_lines_by_quantile import').replace(
                b'from _limen_lines_baseline.', b'from limen.features.')
            assert hashlib.sha256(source).hexdigest() == digest


def _timings(name):
    baseline = getattr(BASELINES[name], name)
    calculate = getattr(importlib.import_module(f'limen.features.{name}'), name)
    frame = DATA.head(64 if name == 'quantile_price_lines' else 512)
    before, after = [], []
    for _ in range(3):
        started = time.perf_counter()
        if name == 'quantile_price_lines':
            expected = pl.concat([baseline(frame.head(rows), 48, .002, include_research_only=False).tail(1)
                                  for rows in range(1, frame.height + 1)])
        else:
            expected = baseline(frame, 48, .002)
        before.append(time.perf_counter() - started)
        started = time.perf_counter()
        result = calculate(frame, 48, .002, **({'include_research_only': False} if name == 'quantile_price_lines' else {}))
        after.append(time.perf_counter() - started)
    assert_frame_equal(result, expected, check_exact=name == 'price_lines', rel_tol=1e-12, abs_tol=1e-12)
    return before, after


@pytest.fixture(scope='module')
def timing_worker():
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn')) as worker:
        yield worker


@pytest.mark.parametrize('name', ['price_lines', 'quantile_price_lines'])
def test_line_latency(name, timing_worker):
    before, after = timing_worker.submit(_timings, name).result(timeout=60)
    assert statistics.median(after) < statistics.median(before) * .5
