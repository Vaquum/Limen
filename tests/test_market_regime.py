from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from limen.features import market_regime
from limen.indicators import sma
from limen.yaml import CompiledSFD, validate
from tests.test_column_shims import _config


FIXTURE = Path(__file__).parent / 'fixtures/spot_1h_20240101_20240116.parquet'


@pytest.fixture(scope='module')
def bars():
    return pl.read_parquet(FIXTURE, n_rows=384)


@pytest.mark.parametrize('lazy', (False, True))
@pytest.mark.parametrize(('short', 'long', 'lookback'), (
    (20, 50, 48), (3, 5, 12), (10, 10, 48), (20, 20, 48), (50, 50, 48),
    (10, 20, 48), (50, 100, 48), (50, 20, 48), (3, 5, 20), (3, 5, 50),
    (12, 26, 12), (12, 26, 26), (20, 50, 20), (20, 50, 50),
))
def test_recorded_market_regime_bindings(bars, lazy, short, long, lookback):
    before = bars.clone()
    source = bars.lazy() if lazy else bars
    result = market_regime(source, short_sma=short, long_sma=long, lookback=lookback)
    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    if lazy:
        result = result.collect()
    short_average = sma(before, 'close', short)[f'sma_{short}']
    long_average = sma(before, 'close', long)[f'sma_{long}']
    volume_average = sma(before, 'volume', lookback)[f'sma_{lookback}']
    expected = {f'sma_{short}': (short_average, short), f'sma_{long}': (long_average, long)}
    expected.setdefault(f'sma_{lookback}', (volume_average, lookback))
    expected.update({'sma_20': (short_average, short), 'sma_50': (long_average, long),
                     'volume_sma': (volume_average, lookback)})
    for name, (average, period) in expected.items():
        assert result.schema[name] == pl.Float64
        np.testing.assert_allclose(result[name].to_numpy(), average.to_numpy(),
                                   rtol=0, atol=0, equal_nan=True)
        assert result[name].null_count() == 0
        assert result[name].is_nan().sum() == period - 1
    warmups = {'trend_strength': max(short, long) - 1,
               'volatility_ratio': max(12, lookback), 'volume_regime': lookback - 1,
               'market_favorable': max(short - 1, long - 1, 12, lookback)}
    for name, count in warmups.items():
        assert result[name].null_count() == count
        assert result[name].is_nan().sum() == 0
    assert len(result.columns) == len(set(result.columns))
    assert set(result.columns) == set(before.columns) | set(expected) | set(warmups)
    assert result.columns[:before.width] == before.columns
    assert_frame_equal(result.select(before.columns), before, check_exact=True)
    assert_frame_equal(source.collect() if lazy else source, before, check_exact=True)
    if (short, long, lookback) == (3, 5, 12):
        assert result.columns == [*before.columns, 'sma_3', 'sma_20', 'sma_5', 'sma_50',
                                  'trend_strength', 'volatility_ratio', 'sma_12',
                                  'volume_sma', 'volume_regime', 'market_favorable']


@pytest.mark.parametrize('lazy', (False, True))
@pytest.mark.parametrize('column', ('sma_20', 'sma_50', 'volume_sma', 'repeat'))
def test_market_regime_refuses_existing_fixed_outputs(bars, lazy, column):
    incoming = (market_regime(bars, short_sma=3, long_sma=5, lookback=12)
                if column == 'repeat' else bars.with_columns(pl.col('close').alias(column)))
    before = incoming.clone()
    with pytest.raises(pl.exceptions.DuplicateError, match='sma_20' if column == 'repeat' else column):
        result = market_regime(incoming.lazy() if lazy else incoming,
                               short_sma=3, long_sma=5, lookback=12)
        if lazy:
            result.collect()
    assert_frame_equal(incoming, before, check_exact=True)


@pytest.mark.parametrize('lazy', (False, True))
def test_market_regime_keeps_generic_sma_replacement(bars, lazy):
    incoming = bars.with_columns(pl.col('volume').alias('sma_3'),
                                 pl.col('close').alias('sma_12'))
    result = market_regime(incoming.lazy() if lazy else incoming,
                           short_sma=3, long_sma=5, lookback=12)
    if lazy:
        result = result.collect()
    for column, source, period in (('sma_3', 'close', 3), ('sma_12', 'volume', 12)):
        np.testing.assert_allclose(result[column].to_numpy(),
                                   sma(bars, source, period)[column].to_numpy(),
                                   rtol=0, atol=0, equal_nan=True)


def test_market_regime_completes_compiled_preparation(bars):
    config = _config()
    declaration = config['sfd']['manifest']
    declaration['data_source']['params']['file_path_or_url'] = str(FIXTURE)
    declaration['features'] = [{'func': 'limen.features.market_regime'}]
    assert validate(config).valid
    prepared = CompiledSFD(config).manifest().prepare_data(bars, {'alpha': 0.3, 'solver': 'svd'})
    for split in ('train', 'val', 'test'):
        features = prepared[f'x_{split}']
        assert features.height > 0
        assert features.height == len(prepared[f'y_{split}'])
        assert features.columns == [name for name in prepared['_feature_names'] if name not in ('datetime', 'next_return')]
        assert {'sma_20', 'sma_50', 'volume_sma'}.issubset(features.columns)
        assert np.isfinite(features.to_numpy()).all()
        assert np.isfinite(prepared[f'y_{split}'].to_numpy()).all()
    prices = prepared['price_data_for_backtest']
    assert prices.height == prepared['x_test'].height
    assert prices['datetime'].is_sorted()
    assert prices['datetime'].n_unique() == prices.height
    observed = bars.join(prices.select('datetime'), on='datetime', how='semi', maintain_order='left')
    assert_frame_equal(prices.select('datetime', 'close'), observed.select('datetime', 'close'),
                       check_exact=True)
