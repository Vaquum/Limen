from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from limen import indicators
from limen.experiment._backtest_provenance import SOURCE_ROW
from limen.indicators.stoch import _stoch_from_arrays
from limen.indicators.stochf import stochf_from_arrays
from limen.indicators.stochrsi import _stochrsi_from_values
from limen.yaml import build_manifest, parse, validate


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tests/fixtures/spot_15m_20250101_20250531.parquet'
PRICES = ('open', 'high', 'low', 'close', 'volume')
DATA = pl.read_parquet(SOURCE).head(512).select('datetime', *PRICES)
COLUMNS = {
    'stoch': ('stoch_slowk', 'stoch_slowd'),
    'stochf': ('stochf_fastk', 'stochf_fastd'),
    'stochrsi': ('stochrsi_fastk', 'stochrsi_fastd'),
}


def _eager_reference(data, name, params):
    # Preserve the old eager extraction/assembly around the unchanged kernels.
    fastk_period = params.get('fastk_period', 5)
    if name == 'stochrsi':
        values = _stochrsi_from_values(
            data[params.get('price_col', 'close')].to_numpy().astype(float, copy=False),
            params.get('period', 14), fastk_period,
            params.get('fastd_period', 3), params.get('fastd_ma_type', 0),
        )
    else:
        high, low, close = (
            data[params.get(f'{column}_col', column)].to_numpy().astype(float, copy=False)
            for column in ('high', 'low', 'close')
        )
        if name == 'stoch':
            values = _stoch_from_arrays(
                high, low, close, fastk_period,
                params.get('slowk_period', 3), params.get('slowk_ma_type', 0),
                params.get('slowd_period', 3), params.get('slowd_ma_type', 0),
            )
        else:
            values = stochf_from_arrays(
                high, low, close, fastk_period,
                params.get('fastd_period', 3), params.get('fastd_ma_type', 0),
            )
    return data.with_columns(pl.Series(column, value)
                             for column, value in zip(COLUMNS[name], values, strict=True))


@pytest.mark.parametrize(('name', 'params', 'rows', 'warmup'), [
    ('stoch', {}, 512, 8),
    ('stochf', {}, 512, 6),
    ('stochrsi', {}, 512, 20),
    ('stoch', {'slowk_ma_type': 3}, 512, 10),
    ('stoch', {'slowd_ma_type': 3}, 512, 10),
    ('stochf', {'fastd_ma_type': 3}, 512, 8),
    ('stochrsi', {'fastd_ma_type': 3}, 512, 22),
    ('stoch', {'high_col': 'close', 'low_col': 'close'}, 512, 8),
    ('stochf', {'high_col': 'close', 'low_col': 'close'}, 512, 6),
    ('stoch', {}, 3, 8),
    ('stochf', {}, 3, 6),
    ('stochrsi', {}, 3, 20),
])
def test_stochastic_recorded_eager_and_lazy_parity(name, params, rows, warmup):
    frame = DATA.head(rows)
    expected = _eager_reference(frame, name, params)
    for source in (frame, frame.lazy()):
        actual = getattr(indicators, name)(source, **params)
        assert isinstance(actual, type(source))
        if isinstance(actual, pl.LazyFrame):
            actual = actual.collect()
        assert_frame_equal(actual, expected, check_exact=True)
        assert actual.columns == frame.columns + list(COLUMNS[name])
        assert actual.select(COLUMNS[name]).dtypes == [pl.Float64, pl.Float64]
        for column in COLUMNS[name]:
            assert actual[column].null_count() == 0
            mask = actual[column].is_nan().to_numpy()
            np.testing.assert_array_equal(mask, np.arange(rows) < warmup)
            assert np.flatnonzero(~mask)[:1].tolist() == ([warmup] if rows > warmup else [])


@pytest.mark.parametrize(('name', 'parameter', 'value'), [
    ('stoch', 'fastk_period', 0),
    ('stoch', 'slowk_period', 0),
    ('stoch', 'slowd_period', 0),
    ('stoch', 'slowk_ma_type', 9),
    ('stoch', 'slowd_ma_type', -1),
    ('stochf', 'fastk_period', 0),
    ('stochf', 'fastd_period', 0),
    ('stochf', 'fastd_ma_type', 9),
    ('stochrsi', 'period', 1),
    ('stochrsi', 'fastk_period', 0),
    ('stochrsi', 'fastd_period', 0),
    ('stochrsi', 'fastd_ma_type', 9),
])
def test_stochastic_rejects_invalid_arguments_before_collection(name, parameter, value):
    for source in (DATA, DATA.lazy()):
        with pytest.raises(ValueError, match=f'{name} {parameter} must be between'):
            getattr(indicators, name)(source, **{parameter: value})


@pytest.mark.parametrize(('name', 'warmup'), [('stoch', 10), ('stochf', 8), ('stochrsi', 22)])
@pytest.mark.parametrize('kind', ['ml', 'rule_based'])
def test_stochastic_yaml_research_and_sensor_match_eager(name, warmup, kind):
    template = 'ridge_regressor' if kind == 'ml' else 'rule_based'
    config, errors = parse(ROOT / f'limen/yaml/templates/{template}.yaml')
    assert not errors
    spec = config['sfd']['manifest']
    spec['data_source'] = {
        'method': 'limen.data.HistoricalData.get_any_file',
        'params': {'file_path_or_url': str(SOURCE), 'columns': ['datetime', *PRICES]},
    }
    spec['split_dates'] = dict(zip(
        ('train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end'),
        ('2025-01-01', '2025-01-03', '2025-01-03', '2025-01-04', '2025-01-04', '2025-01-07'),
        strict=True,
    ))
    spec['indicators'] = [{'func': f'limen.indicators.{name}',
                           'params': {'fastk_period': '{fastk_period}'}}]
    spec['backtest'] = {'take_profit_bps': 50.0}
    config['sfd']['params'] = {'fastk_period': [7]}
    config['uel']['n_permutations'] = 1
    if kind == 'ml':
        del spec['features'], spec['scaler']
    else:
        spec['strategy'] = {
            'conditions': [{'id': 'entry', 'name': 'Stochastic threshold',
                            'type': 'threshold', 'column': COLUMNS[name][0],
                            'operator': '>', 'value': 50.0}],
            'entry': 'entry',
        }
    report = validate(config)
    assert report.valid, report.errors
    manifest = build_manifest(config)
    reference = build_manifest(config)
    reference.feature_transforms[0].func = (
        lambda frame, **params: _eager_reference(frame.collect(), name, params).lazy()
    )
    params = {'fastk_period': 7}
    assert manifest.resolve_trade_policy(params) is None
    actual = manifest.prepare_data(DATA, params)
    expected = reference.prepare_data(DATA, params)
    witness = actual['_backtest_provenance']
    assert witness == expected['_backtest_provenance']
    assert actual['_alignment'] == expected['_alignment']
    for split, proof, offset, count in zip(
        ('train', 'val', 'test'), witness.splits, (0, 192, 288), (192, 96, 224), strict=True,
    ):
        assert proof.error is None
        assert (proof.source_offset, proof.source_count) == (offset, count)
        start = offset + (warmup if kind == 'rule_based' or split == 'train' else 0)
        stop = offset + count - (kind == 'ml')
        assert proof.retained == range(start, stop)
        key = f'x_{split}' if kind == 'ml' else split
        assert_frame_equal(actual[key], expected[key], check_exact=True)
        assert SOURCE_ROW not in actual[key].columns
        assert_frame_equal(actual[key].select(PRICES), DATA.slice(start, stop - start).select(PRICES), check_exact=True)
        if kind == 'ml':
            assert_series_equal(actual[f'y_{split}'], expected[f'y_{split}'], check_exact=True)
        else:
            np.testing.assert_array_equal(actual[f'x_{split}'], expected[f'x_{split}'])
    if kind == 'ml':
        assert actual['_feature_names'] == expected['_feature_names']
        assert SOURCE_ROW not in actual['_feature_names']
        assert_frame_equal(actual['price_data_for_backtest'], expected['price_data_for_backtest'], check_exact=True)
    sensor, lookback = manifest.sensor_input_prep(DATA, actual['_fitted_params'], params)
    eager_sensor, eager_lookback = reference.sensor_input_prep(DATA, expected['_fitted_params'], params)
    assert_frame_equal(sensor, eager_sensor, check_exact=True)
    assert lookback == eager_lookback == warmup
    assert sensor.height == DATA.height
    assert SOURCE_ROW not in sensor.columns
