"""Baseline contracts and execution regressions on recorded BTCUSDT bars."""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from limen import indicators

FIXTURES = Path(__file__).parent / 'fixtures'
REFERENCE = json.loads((FIXTURES / 'indicator_equivalence.json').read_text())
SOURCE = FIXTURES.parent.parent / REFERENCE['input']
DATA = pl.read_parquet(SOURCE).select('open', 'high', 'low', 'close', 'volume')


def _case_frame(case):
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
        return frame.with_columns(pl.when(pl.int_range(pl.len()) == index)
                                  .then(pl.lit(missing, dtype=pl.Float64))
                                  .otherwise(pl.col('close')).alias('close'))
    return frame


@pytest.mark.parametrize('case', REFERENCE['cases'], ids=lambda c: f'{c["name"]}-{c["variant"]}-{c["rows"]}-{c["kwargs"]}')
def test_baseline_equivalence(case):
    function = getattr(indicators, case['name'])
    assert str(inspect.signature(function)) == REFERENCE['signatures'][case['name']]
    frame = _case_frame(case)
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
            with np.load(FIXTURES / 'indicator_equivalence.npz') as expected:
                for column in case['columns']:
                    assert result[column].null_count() == 0
                    np.testing.assert_array_equal(result[column].to_numpy(),
                                                  expected[f'{case["id"]}:{column}'])


def test_fixture_provenance():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == REFERENCE['input_sha256']


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


def _scalar_cci(data, period):
    price = (data['high'].to_numpy() + data['low'].to_numpy() + data['close'].to_numpy()) / 3.0
    result = np.full(len(price), np.nan)
    for index in range(period - 1, len(price)):
        window = price[index - period + 1:index + 1]
        average = 0.0
        for value in window:
            average += value
        average /= period
        deviation = 0.0
        for value in window:
            deviation += abs(value - average)
        difference = price[index] - average
        result[index] = difference / (0.015 * (deviation / period)) if difference and deviation else 0.0
    return result


def test_native_window_latency():
    frame = DATA.head(6000)
    started = time.perf_counter()
    expected = _scalar_cci(frame, 14)
    scalar_time = time.perf_counter() - started
    started = time.perf_counter()
    result = indicators.cci(frame)['cci_14'].to_numpy()
    native_time = time.perf_counter() - started
    np.testing.assert_array_equal(result, expected)
    assert native_time < scalar_time * 0.5
