from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import polars as pl
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from limen import Cohort, Sensor
from limen.experiment.holdout import freeze_holdout_selection
from limen.inference.sensor import BarPrediction


FIXTURE = Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet'
CUTOFF = datetime(2025, 1, 16, tzinfo=timezone.utc)


@pytest.fixture(scope='module')
def recorded():
    return pl.read_parquet(FIXTURE, n_rows=1500)


def _metadata(recorded):
    return recorded[1440:1443].select('datetime').with_columns(
        (pl.col('datetime').dt.epoch('ns') + 900_000_000_000).alias('available_at_ns')
    )


def _sensor(metadata, pairs=None, **changes):
    values = [(1.25, None)] * metadata.height if pairs is None else pairs
    options = {
        'fitted_state': object(), 'params': {},
        'infer': lambda history, state, params: (metadata, values),
        'map_output': lambda output: output, 'state_id': str(uuid4()),
        'fitting_cutoff': CUTOFF - timedelta(days=1),
        'selection_cutoff': CUTOFF, 'output_id': 'next_return',
    }
    options.update(changes)
    return Sensor.from_python(**options)


def test_python_cohort_defaults_and_identity(recorded):
    metadata = _metadata(recorded)
    first = _sensor(metadata, [(-2.0, 0.2), (0.0, 0.4), (4.0, 0.6)])
    second = _sensor(metadata, [(1.0, 0.8), (2.0, None), (-2.0, 0.2)])
    supplied = [first, second]
    cohort = Cohort.from_python(supplied)
    supplied.clear()
    assert isinstance(cohort, Cohort)
    assert cohort.members == (first, second)
    assert cohort.members[0] is first and cohort.members[1] is second
    assert UUID(cohort.state_id).version == 4
    assert cohort.origin == cohort.prediction_mode == 'python'
    assert cohort.manifest_id is cohort.cohort_id is None
    assert cohort.selection_cutoff == CUTOFF
    rows = cohort(recorded)
    assert [row.prediction for row in rows] == [-0.5, 1.0, 1.0]
    assert [row.probability for row in rows] == [0.5, None, 0.4]
    assert all(row.reason is None and row.trade_contract_digest is None for row in rows)
    assert cohort.predict(recorded) == rows[-1]
    assert cohort.predict_all(recorded) == rows
    single = Cohort.from_python([first])
    assert single(recorded) == first(recorded)
    for name in ('members', 'state_id', 'selection_cutoff', 'manifest_id', 'cohort_id'):
        with pytest.raises(FrozenInstanceError):
            setattr(cohort, name, None)
    for name in ('origin', 'prediction_mode'):
        with pytest.raises(AttributeError):
            setattr(cohort, name, 'compiled')
    for name in (
        'output_id', 'member_ids', 'aggregation_mode', 'architecture_id',
        'permutation_ids', 'available_permutation_ids', 'supports_probabilities',
        'metadata', 'experiment_id', 'experiment_dir', 'source_metadata',
    ):
        with pytest.raises(AttributeError):
            getattr(cohort, name)
    identity = cohort.state_id
    for replacement in ([second], object()):
        with pytest.raises(TypeError, match='membership is fixed'):
            cohort.set_members(replacement)
    assert cohort.state_id == identity and cohort.members == (first, second)
    scenarios = [{
        'fee_bps': 0.0, 'slip_bps': 0.0, 'notional_rate': 1.0,
        'execution_lag_bars': 1, 'take_profit_bps': None, 'stop_loss_bps': None,
    }]
    for selected in (first, cohort):
        with patch.object(type(first), 'predict_all', side_effect=AssertionError('must reject before inference')):
            with pytest.raises(ValueError, match='Holdout accepts exact Sensor or Cohort only'):
                freeze_holdout_selection(selected, context_start='2025-01-01', scenarios=scenarios)


def test_python_cohort_selection_boundaries(recorded):
    metadata = _metadata(recorded)
    stamps = metadata['datetime'].to_list()
    first = _sensor(metadata)
    second = _sensor(metadata, selection_cutoff=stamps[1])
    called = []

    def aggregate(bars):
        called.append(bars[0].datetime)
        return sum(bar.prediction for bar in bars), None

    cohort = Cohort.from_python([first, second], aggregate=aggregate)
    rows = cohort(recorded)
    assert cohort.selection_cutoff == stamps[1]
    assert [row.reason for row in rows] == ['inside-training-window', None, None]
    assert called == stamps[1:]
    assert rows[1].prediction == 2.5
    called.clear()
    added = Cohort.from_python([first, second], aggregate=aggregate, selection_cutoff=stamps[2])
    assert [row.reason for row in added(recorded)] == ['inside-training-window', 'inside-training-window', None]
    assert called == stamps[2:]
    assert added.selection_cutoff == stamps[2]
    earlier = Cohort.from_python([first, second], selection_cutoff=CUTOFF - timedelta(hours=1))
    assert earlier.selection_cutoff == stamps[1]


def test_python_cohort_output_compatibility(recorded):
    metadata = _metadata(recorded)
    first = _sensor(metadata)
    compatible = _sensor(metadata)
    assert Cohort.from_python([first, compatible])(recorded)[0].prediction == 1.25
    for other in (_sensor(metadata, output_id=None), _sensor(metadata, output_id='return_bps')):
        with pytest.raises(ValueError, match='matching declared'):
            Cohort.from_python([first, other])
    with pytest.raises(ValueError, match='requires map_output'):
        Cohort.from_python([first], mapped_output_id='relabelled')
    with pytest.raises(ValueError, match='requires mapped_output_id'):
        Cohort.from_python([first], map_output=lambda sensor, bar: (bar.prediction, bar.probability))
    bps = _sensor(metadata, [(12500.0, None)] * metadata.height, output_id='return_bps')

    def common_units(sensor, bar):
        return bar.prediction / 10000 if sensor is bps else bar.prediction, bar.probability

    mapped = Cohort.from_python([first, bps], map_output=common_units, mapped_output_id='next_return')
    assert [row.prediction for row in mapped(recorded)] == [1.25] * metadata.height
    # Custom aggregation owns final semantics; input-domain metadata cannot claim its output units.
    signed = Cohort.from_python([first], aggregate=lambda bars: (int(bars[0].prediction > 0), None))
    assert signed(recorded)[0].prediction == 1
    assert not hasattr(signed, 'output_id')


def test_python_cohort_alignment_availability_and_blocking(recorded):
    metadata = _metadata(recorded)
    first = _sensor(metadata)
    later = metadata.with_columns((pl.col('available_at_ns') + 1).alias('available_at_ns'))
    rows = Cohort.from_python([first, _sensor(later)])(recorded)
    assert [row.available_at_ns for row in rows] == later['available_at_ns'].to_list()
    unknown = _sensor(metadata.drop('available_at_ns'))
    assert all(row.available_at_ns is None for row in Cohort.from_python([first, unknown])(recorded))
    for misaligned in (metadata[1:], metadata.head(0)):
        with pytest.raises(ValueError, match='member 1.*do not align'):
            Cohort.from_python([first, _sensor(misaligned)])(recorded)
    empty = Cohort.from_python([_sensor(metadata.head(0)), _sensor(metadata.head(0))])
    assert empty(recorded) == []
    assert empty.predict(recorded) == BarPrediction(None, None, None, 'warm-up')
    calls = []

    def forbidden(*args):
        calls.append(args)
        raise AssertionError('blocked rows must not enter mapping or aggregation')

    blocked = metadata.with_columns(pl.Series('reason', ['warm-up', 'null-features', 'warm-up']))
    malformed = _sensor(metadata, [(float('nan'), None)] * metadata.height)
    cohort = Cohort.from_python(
        [_sensor(blocked), malformed], aggregate=forbidden,
        map_output=forbidden, mapped_output_id='next_return',
        selection_cutoff=metadata['datetime'][1],
    )
    assert [row.reason for row in cohort(recorded)] == ['inside-training-window', 'sensor-error', 'sensor-error']
    assert calls == []
    pure = Cohort.from_python([_sensor(blocked), _sensor(metadata)])
    assert [row.reason for row in pure(recorded)] == ['warm-up', 'null-features', 'warm-up']
    baseline = first(recorded)
    malformed_batches = [
        [replace(baseline[0], datetime=None)],
        [replace(baseline[0], datetime=baseline[0].datetime.replace(tzinfo=None))],
        [baseline[0], baseline[0]],
        baseline[::-1],
        [replace(baseline[0], reason='unknown')],
        [replace(baseline[0], available_at_ns=True)],
        [replace(baseline[0], available_at_ns=0)],
        [object()], 'bad batch',
    ]
    for batch in malformed_batches:
        with patch.object(type(first), 'predict_all', return_value=batch):
            with pytest.raises(ValueError, match='member 0'):
                Cohort.from_python([first])(recorded)
    for pair in ((True, None), (float('inf'), None), (1.0, 1.1)):
        invalid = replace(baseline[0], prediction=pair[0], probability=pair[1])
        with patch.object(type(first), 'predict_all', return_value=[invalid]):
            row = Cohort.from_python([first])(recorded)[0]
        assert row.reason == 'sensor-error' and row.prediction is row.probability is None


def test_python_cohort_custom_callbacks_and_failures(recorded):
    metadata = _metadata(recorded)
    state = object()
    history = {'bars': recorded, 'arbitrary': object()}
    inference_calls = []

    def infer(supplied, fitted, params):
        assert supplied is history['bars'] and fitted is state and params == {}
        inference_calls.append(supplied)
        return metadata, [(2.0, 0.4)] * metadata.height

    member = _sensor(metadata, fitted_state=state, infer=infer)
    original = member(recorded)
    preparation_calls = []

    def prepare(supplied, sensor):
        assert supplied is history and sensor is member
        preparation_calls.append(sensor)
        return supplied['bars']

    def mapper(sensor, bar):
        assert sensor is member
        bar.datetime = None
        bar.reason = 'sensor-error'
        bar.available_at_ns = 0
        bar.trade_contract_digest = 'changed'
        return -3.0, 0.6

    def aggregate(bars):
        assert bars[0].prediction == -3.0 and bars[0].reason is None
        bars[0].datetime = None
        bars[0].available_at_ns = 0
        return -7.0, 0.7

    cohort = Cohort.from_python(
        [member], prepare=prepare, map_output=mapper,
        mapped_output_id='mapped_signal', aggregate=aggregate,
    )
    inference_calls.clear()
    rows = cohort(history)
    assert preparation_calls == [member] and inference_calls == [recorded]
    assert [row.prediction for row in rows] == [-7.0] * metadata.height
    assert all(row.probability == 0.7 and row.reason is None for row in rows)
    assert [row.datetime for row in rows] == metadata['datetime'].to_list()
    assert [row.available_at_ns for row in rows] == metadata['available_at_ns'].to_list()
    assert all(row.trade_contract_digest is None for row in rows)
    assert member(recorded) == original
    for invalid in ((True, None), (float('nan'), None), (2.0, -0.1), (None, None), 'invalid'):
        mapped = Cohort.from_python([member], map_output=lambda sensor, bar: invalid, mapped_output_id='mapped')
        assert all(row.reason == 'sensor-error' for row in mapped(recorded))
        aggregated = Cohort.from_python([member], aggregate=lambda bars: invalid)
        assert all(row.reason == 'sensor-error' for row in aggregated(recorded))

    def fail(*args):
        raise RuntimeError('callback failed')

    cases = ({'prepare': fail}, {'aggregate': fail}, {'map_output': fail, 'mapped_output_id': 'mapped'})
    for options in cases:
        with pytest.raises(RuntimeError, match='callback failed'):
            Cohort.from_python([member], **options)(recorded)
    failing = _sensor(metadata, infer=fail)
    with pytest.raises(RuntimeError, match='callback failed'):
        Cohort.from_python([failing])(recorded)


def test_python_cohort_rejects_invalid_construction(recorded):
    metadata = _metadata(recorded)
    first = _sensor(metadata, state_id='explicit-state')
    repeated = _sensor(metadata, state_id='explicit-state')
    cohort = Cohort.from_python([first])
    cases = [
        (TypeError, first, {}), (TypeError, iter([first]), {}),
        (TypeError, {first}, {}), (TypeError, [cohort], {}),
        (TypeError, [object()], {}), (ValueError, [], {}),
        (ValueError, [first, first], {}), (ValueError, [first, repeated], {}),
        (TypeError, [first], {'aggregate': None}),
        (ValueError, [first], {'aggregate': 'median'}),
        (TypeError, [first], {'prepare': 1}), (TypeError, [first], {'map_output': 1}),
        (TypeError, [first], {'state_id': 1}), (ValueError, [first], {'state_id': ' '}),
        (TypeError, [first], {'selection_cutoff': '2025-01-16'}),
        (ValueError, [first], {'selection_cutoff': CUTOFF.replace(tzinfo=None)}),
        (TypeError, [first], {'map_output': lambda sensor, bar: (1.0, None), 'mapped_output_id': 1}),
        (ValueError, [first], {'map_output': lambda sensor, bar: (1.0, None), 'mapped_output_id': ' '}),
    ]
    with patch.object(type(first), 'predict_all', side_effect=AssertionError('construction must not infer')):
        for error, members, options in cases:
            with pytest.raises(error):
                Cohort.from_python(members, **options)
    assert Cohort.from_python([first], state_id=' chosen ').state_id == ' chosen '


def test_python_cohort_classification_vote(recorded):
    def prepare(history, params):
        return history.select(
            'datetime',
            pl.col('datetime').dt.strftime('%A').alias('weekday'),
            (pl.col('close') / pl.col('close').shift(4) - 1).alias('momentum'),
        )

    features = prepare(recorded[:960], {}).with_columns(
        (recorded[:960]['close'].shift(-1) / recorded[:960]['open'].shift(-1) - 1 > 0).alias('target')
    ).drop_nulls()
    columns = ['weekday', 'momentum']
    with pytest.raises(ValueError):
        float(features['weekday'][0])
    model = Pipeline([
        ('features', ColumnTransformer([
            ('weekday', OneHotEncoder(handle_unknown='ignore'), ['weekday']),
            ('momentum', StandardScaler(), ['momentum']),
        ])),
        ('classifier', LogisticRegression(max_iter=200)),
    ]).fit(features.select(columns), features['target'])
    assert list(model.feature_names_in_) == columns
    history = recorded[1440:1452]
    sensor = Sensor.from_model(model, selection_cutoff=CUTOFF, prepare=prepare, output_id='up_class')
    cohort = Cohort.from_python([sensor], aggregate='vote')
    rows = cohort(history)
    expected = model.predict(prepare(history, {}).drop_nulls().select(columns))
    assert [row.reason for row in rows[:4]] == ['warm-up'] * 4
    assert [row.prediction for row in rows[4:]] == [int(value) for value in expected]
    assert all(type(row.prediction) is int and row.probability is None for row in rows[4:])
    seen = []

    def predict(fitted, frame, params):
        seen.append(frame.schema)
        return fitted.predict(frame)

    custom = Sensor.from_model(
        model, selection_cutoff=CUTOFF, prepare=prepare, predict=predict, output_id='up_class',
    )
    assert Cohort.from_python([custom], aggregate='vote')(history) == rows
    assert seen == [pl.Schema({'weekday': pl.String, 'momentum': pl.Float64})]
    metadata = _metadata(recorded)
    zero = _sensor(metadata, [(0, 0.4)] * metadata.height, output_id='up_class')
    one = _sensor(metadata, [(1.0, 0.8)] * metadata.height, output_id='up_class')
    assert all(row.prediction == 0 and row.probability is None for row in Cohort.from_python([zero, one], aggregate='vote')(recorded))
    for value in (2.0, 0.25):
        invalid = _sensor(metadata, [(value, None)] * metadata.height, output_id='up_class')
        assert all(row.reason == 'sensor-error' for row in Cohort.from_python([invalid], aggregate='vote')(recorded))
    with pytest.raises(ValueError, match='Holdout accepts exact Sensor or Cohort only'):
        freeze_holdout_selection(
            sensor, context_start='2025-01-01', scenarios=[{
                'fee_bps': 0.0, 'slip_bps': 0.0, 'notional_rate': 1.0,
                'execution_lag_bars': 1, 'take_profit_bps': None, 'stop_loss_bps': None,
            }],
        )
