from dataclasses import FrozenInstanceError
from datetime import timedelta
from pathlib import Path
from uuid import UUID

import numpy as np
import polars as pl
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from limen import Sensor


FIXTURE = Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet'


@pytest.fixture(scope='module')
def recorded_model():
    recorded = pl.read_parquet(FIXTURE, n_rows=1500)
    features = recorded.select(
        'datetime',
        (pl.col('close') / pl.col('open') - 1).alias('momentum'),
        ((pl.col('high') - pl.col('low')) / pl.col('open')).alias('span'),
        (pl.col('close').shift(-1) / pl.col('open').shift(-1) - 1).alias('next_return'),
        (pl.col('datetime').dt.epoch('ns') + 900_000_000_000).alias('available_at_ns'),
    )
    training = features[:960]
    columns = ['span', 'momentum']
    named = Ridge().fit(training.select(columns), training['next_return'])
    unnamed = Ridge().fit(training.select(columns).to_numpy(), training['next_return'])
    imputer = Pipeline([('imputer', SimpleImputer()), ('model', Ridge())]).fit(
        training.select(columns), training['next_return'],
    )
    return named, unnamed, imputer, features[1440:1446].drop('next_return')


def test_from_model_defaults_and_feature_selection(recorded_model):
    named, unnamed, _imputer, history = recorded_model
    cutoff = history['datetime'][0]
    sensor = Sensor.from_model(named, selection_cutoff=cutoff)
    assert isinstance(sensor, Sensor)
    assert sensor.fitted_state is named
    assert sensor.params == {}
    assert UUID(sensor.state_id).version == 4
    assert sensor.fitting_cutoff == sensor.selection_cutoff == cutoff
    assert sensor.output_id is None
    assert sensor.manifest_id is sensor.permutation_id is None
    expected = named.predict(history.select('span', 'momentum'))
    results = sensor(history)
    np.testing.assert_allclose([row.prediction for row in results], expected)
    assert all(row.probability is None and row.reason is None for row in results)
    assert [row.available_at_ns for row in results] == history['available_at_ns'].to_list()
    assert all(row.trade_contract_digest is None for row in results)
    assert sensor.predict(history) == results[-1]
    assert sensor.predict_all(history.head(0)) == []
    assert sensor.predict(history.head(0)).reason == 'warm-up'
    with pytest.raises(FrozenInstanceError):
        sensor.output_id = 'changed'
    params = {'nested': []}
    columns = ['span', 'momentum']
    explicit = Sensor.from_model(
        unnamed, feature_columns=columns, params=params,
        selection_cutoff=cutoff, fitting_cutoff=cutoff - timedelta(days=1),
        state_id=' chosen ', output_id=' next_return ',
    )
    columns.reverse()
    params['later'] = 1
    assert explicit.state_id == ' chosen '
    assert explicit.output_id == ' next_return '
    assert 'later' not in explicit.params
    assert explicit.params['nested'] is params['nested']
    with pytest.raises(TypeError):
        explicit.params['changed'] = 1
    np.testing.assert_allclose([row.prediction for row in explicit(history)], expected)
    fallback = Sensor.from_model(unnamed, selection_cutoff=cutoff)
    np.testing.assert_allclose(
        [row.prediction for row in fallback(history.select('datetime', 'span', 'momentum'))],
        expected,
    )


def test_from_model_custom_hooks_and_unavailability(recorded_model):
    named, _unnamed, imputer, history = recorded_model
    cutoff = history['datetime'][1]
    unavailable = history.with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(None)
        .when(pl.int_range(pl.len()) == 2).then(float('nan'))
        .when(pl.int_range(pl.len()) == 3).then(float('inf'))
        .otherwise(pl.col('momentum')).alias('momentum'),
        pl.Series('reason', [None, None, None, None, 'warm-up', None]),
    )
    calls = []
    mapped_calls = []

    def prepare(supplied, params):
        assert params == {'scale': 10000}
        return supplied['bars']

    def predict(model, frame, params):
        assert model is named
        assert frame.columns == ['span', 'momentum']
        assert frame.schema == {'span': pl.Float64, 'momentum': pl.Float64}
        assert params == {'scale': 10000}
        calls.append(frame)
        return frame['momentum']

    def map_output(output):
        mapped_calls.append(output)
        return [(value * 10000, 0.6) for value in output]

    sensor = Sensor.from_model(
        named, prepare=prepare, predict=predict, map_output=map_output,
        params={'scale': 10000}, selection_cutoff=cutoff,
    )
    results = sensor({'bars': unavailable})
    assert [row.reason for row in results] == [
        'inside-training-window', None, 'null-features', 'null-features', 'warm-up', None,
    ]
    assert len(calls) == 1
    assert calls[0].equals(history[[1, 5]].select('span', 'momentum'))
    assert results[1].prediction == history['momentum'][1] * 10000
    assert results[1].probability == 0.6
    blocked = unavailable.with_columns(pl.lit('warm-up').alias('reason'))
    assert all(row.reason is not None for row in sensor({'bars': blocked}))
    assert len(calls) == len(mapped_calls) == 1
    first_cutoff = history['datetime'][0]
    missing = history.with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col('momentum')).alias('momentum'),
    )
    assert Sensor.from_model(named, selection_cutoff=first_cutoff)(missing)[0].reason == 'warm-up'
    assert Sensor.from_model(named, selection_cutoff=first_cutoff)(
        history.with_columns(pl.lit(None, dtype=pl.Float64).alias('momentum')),
    )[0].reason == 'warm-up'
    forwarding = Sensor.from_model(imputer, missing='model', selection_cutoff=first_cutoff)
    expected = imputer.predict(missing.select('span', 'momentum'))
    np.testing.assert_allclose([row.prediction for row in forwarding(missing)], expected)
    assert all(row.reason is None for row in forwarding(missing))
    for callback in ('prepare', 'predict', 'map_output'):
        def fail(*args):
            raise RuntimeError('caller failure')
        with pytest.raises(RuntimeError, match='caller failure'):
            Sensor.from_model(named, selection_cutoff=first_cutoff, **{callback: fail})(history)


def test_from_model_rejects_invalid_construction_and_shapes(recorded_model):
    named, _unnamed, _imputer, history = recorded_model
    cutoff = history['datetime'][0]
    for argument, value in (
        ('prepare', 1), ('predict', 1), ('map_output', 1), ('params', []),
        ('params', {1: 1}), ('state_id', 1), ('output_id', 1),
        ('feature_columns', 'momentum'), ('feature_columns', b'momentum'),
        ('feature_columns', [1]), ('feature_columns', iter(['momentum'])),
    ):
        with pytest.raises(TypeError):
            Sensor.from_model(named, selection_cutoff=cutoff, **{argument: value})
    for argument, value in (
        ('state_id', ' '), ('output_id', ''), ('feature_columns', []),
        ('feature_columns', ['momentum', 'momentum']), ('feature_columns', [' ']),
        ('missing', 'ignore'), ('model_input', 'dict'),
        ('fitting_cutoff', cutoff + timedelta(seconds=1)),
        ('selection_cutoff', cutoff.replace(tzinfo=None)),
    ):
        with pytest.raises(ValueError):
            Sensor.from_model(named, **({'selection_cutoff': cutoff} | {argument: value}))
    with pytest.raises(TypeError, match='callable predict'):
        Sensor.from_model(object(), selection_cutoff=cutoff)
    with pytest.raises(TypeError, match='datetimes'):
        Sensor.from_model(named, selection_cutoff='2025-01-16')
    with pytest.raises(ValueError, match='absent'):
        Sensor.from_model(named, feature_columns=['absent'], selection_cutoff=cutoff)(history)
    for prepared in (
        history.drop('datetime'),
        history.with_columns(pl.col('datetime').dt.replace_time_zone(None)),
        pl.concat([history, history.head(1)]),
        history.with_columns(pl.lit('sensor-error').alias('reason')),
        history.with_columns((pl.col('datetime').dt.epoch('ns') - 1).alias('available_at_ns')),
    ):
        with pytest.raises(ValueError):
            Sensor.from_model(named, selection_cutoff=cutoff)(prepared)
    for supplied in ([], {'bars': history}):
        with pytest.raises(TypeError, match='Polars DataFrame'):
            Sensor.from_model(named, selection_cutoff=cutoff)(supplied)
    with pytest.raises(TypeError, match='Polars DataFrame'):
        Sensor.from_model(named, prepare=lambda h, p: [], selection_cutoff=cutoff)(history)
    for output in (
        history['momentum'].to_numpy().reshape(-1, 1), [[value] for value in history['momentum']],
        history['momentum'].head(1), 'invalid', 1, iter(history['momentum']),
    ):
        with pytest.raises(ValueError):
            Sensor.from_model(named, predict=lambda m, f, p, output=output: output, selection_cutoff=cutoff)(history)
    with pytest.raises(ValueError, match='row count'):
        Sensor.from_model(
            named, map_output=lambda output: [], selection_cutoff=cutoff,
        )(history)


def test_from_model_boolean_containers_and_invalid_aligned_scalars(recorded_model):
    named, _unnamed, _imputer, history = recorded_model
    cutoff = history['datetime'][0]
    labels = (history['momentum'] > 0).to_list()
    for output in (
        labels, tuple(labels), np.asarray(labels), pl.Series(labels),
        np.asarray([np.bool_(value) for value in labels], dtype=object),
    ):
        sensor = Sensor.from_model(named, predict=lambda m, f, p, output=output: output, selection_cutoff=cutoff)
        results = sensor(history)
        assert [row.prediction for row in results] == [int(value) for value in labels]
        assert all(type(row.prediction) is int and row.reason is None for row in results)
    for invalid in (float('nan'), float('inf'), None, object(), 'bad', complex(1, 2)):
        sensor = Sensor.from_model(
            named, predict=lambda m, f, p, invalid=invalid: [invalid] * f.height, selection_cutoff=cutoff,
        )
        assert all(row.reason == 'sensor-error' and row.prediction is row.probability is None for row in sensor(history))
    advanced = Sensor.from_python(
        fitted_state=named, params={}, state_id='advanced-bools',
        fitting_cutoff=cutoff, selection_cutoff=cutoff,
        infer=lambda h, m, p: (h.select('datetime'), labels),
        map_output=lambda output: [(value, None) for value in output],
    )
    assert all(row.reason == 'sensor-error' for row in advanced(history))


def test_from_model_explicit_model_input_and_custom_prediction(recorded_model):
    _named, _unnamed, _imputer, history = recorded_model
    cutoff = history['datetime'][0]

    class FrameModel:
        def predict(self, frame):
            assert isinstance(frame, pl.DataFrame)
            return frame['momentum']

    class ArrayModel:
        def predict(self, array):
            assert isinstance(array, np.ndarray)
            return array[:, 0]

    for model, mode in ((FrameModel(), 'frame'), (ArrayModel(), 'array'), (ArrayModel(), 'auto')):
        sensor = Sensor.from_model(
            model, feature_columns=['momentum'], model_input=mode, selection_cutoff=cutoff,
        )
        assert [row.prediction for row in sensor(history)] == history['momentum'].to_list()
    sensor = Sensor.from_model(
        object(), feature_columns=['momentum'], model_input='array',
        predict=lambda m, f, p: f['momentum'], selection_cutoff=cutoff,
    )
    assert [row.prediction for row in sensor(history)] == history['momentum'].to_list()
    categorical = history.with_columns(
        pl.when(pl.int_range(pl.len()) == 2).then(None)
        .otherwise(pl.col('datetime').dt.strftime('%A')).alias('weekday'),
    )
    sensor = Sensor.from_model(
        object(), feature_columns=['weekday', 'momentum'],
        predict=lambda m, f, p: f['momentum'], selection_cutoff=cutoff,
    )
    assert [row.reason for row in sensor(categorical)] == [None, None, 'null-features', None, None, None]
