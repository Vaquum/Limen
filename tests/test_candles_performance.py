"""Candlestick equivalence and scalar-dispatch costs on recorded market bars."""
import ast
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
    frame = DATA.head(512 if variant == 'raw' else 128)
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
    frame = DATA.head(1500)
    before, after = [], []
    iterations = 1
    for _ in range(3):
        while True:
            started = time.process_time()
            for _ in range(iterations):
                expected = BASELINES[name](frame)
            elapsed = time.process_time() - started
            if elapsed >= .03:
                break
            iterations *= 2
        before.append(elapsed / iterations)
        started = time.process_time()
        for _ in range(iterations):
            result = getattr(indicators, name)(frame)
        after.append((time.process_time() - started) / iterations)
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


class _AllowedRewrites(ast.NodeTransformer):
    def visit_Import(self, node):
        if len(node.names) == 1 and node.names[0].name == 'limen._feature_constraints':
            return None
        return node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) in ('_check.names', '_check.numbers'):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'tolist' and not node.args and not node.keywords:
                return node.func.value
            if ast.unparse(node.func) == 'np.float64' and len(node.args) == 1:
                return node.args[0]
        return node

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.op, ast.Mult) and isinstance(node.left, ast.List)
                and ast.literal_eval(node.left) == [0.0]):
            return ast.parse(f'np.zeros({ast.unparse(node.right)}, dtype=float)').body[0].value
        return node


def test_structural_equivalence():
    root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(ARCHIVE) as saved:
        for path in REFERENCE['sources']:
            before = ast.parse(saved.read('_limen_candles_baseline/' + Path(path).name))
            after = _AllowedRewrites().visit(ast.parse((root / path).read_text()))
            assert ast.dump(after) == ast.dump(before), path


@pytest.mark.parametrize('name', ['cdlabandonedbaby', 'cdldarkcloudcover', 'cdlmathold'])
def test_numpy_penetration(name):
    from limen import indicators
    frame = pl.read_parquet(FIXTURES / 'indicator_equivalence_long.parquet').head(5444)
    for penetration in (np.float16(.5), np.float32(.5570075), np.float64(.5), .2, .8):
        with np.errstate(over='ignore'):
            expected = BASELINES[name](frame, penetration=penetration)
            assert_frame_equal(getattr(indicators, name)(frame, penetration=penetration), expected, check_exact=True)


@pytest.mark.parametrize('name', ['cdldoji', 'cdlhammer', 'cdllongline'])
def test_infinite_values(name):
    from limen import indicators
    frame = DATA.head(128).with_columns(
        pl.when(pl.int_range(pl.len()).is_in([20, 30])).then(pl.lit(np.inf))
        .when(pl.int_range(pl.len()) == 40).then(pl.lit(-np.inf))
        .otherwise(pl.col('high')).alias('high'))
    with np.errstate(invalid='ignore'):
        expected = BASELINES[name](frame)
        assert_frame_equal(getattr(indicators, name)(frame), expected, check_exact=True)
