from datetime import timedelta
from math import fsum
from pathlib import Path

import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from limen import Sensor
from limen.cohort import bayesian_cohort


@pytest.fixture(scope='module')
def recorded():
    source = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'
    bars = (pl.read_parquet(source, n_rows=1100)
            .with_columns(pl.col('close').pct_change().alias('return_1'),
                          pl.col('close').pct_change(3).alias('return_3'),
                          (pl.col('close').shift(-1) > pl.col('close')).cast(pl.Int8).alias('outcome'))
            .drop_nulls())
    features = ['return_1', 'return_3']
    cutoff = bars['datetime'][502]
    models = [LogisticRegression(C=10), DecisionTreeClassifier(max_depth=3, random_state=42)]
    sensors = []
    for index, model in enumerate(models):
        model.fit(bars[:500].select(features).to_numpy(), bars[:500]['outcome'].to_numpy())
        sensors.append(Sensor.from_model(model, selection_cutoff=cutoff, feature_columns=features,
                                        state_id=f'recorded-fit-{index}', output_id='next_close_up'))
    evidence = []
    for sensor in sensors:
        predictions = sensor.predict_all(bars[510:640])
        for bar, outcome in zip(predictions, bars[510:640]['outcome'], strict=True):
            assert bar.reason is None
            evidence.append({'sensor_id': sensor.state_id, 'datetime': bar.datetime,
                             'available_at': bar.datetime + timedelta(hours=2),
                             'prediction': bar.prediction, 'outcome': outcome,
                             'source': str(source) + '#next-hour-close'})
    return bars, sensors, pl.DataFrame(evidence)


def test_recorded_bayesian_cohort(recorded):
    bars, sensors, evidence = recorded
    alpha, beta = 2.0, 3.0
    cohort = bayesian_cohort(sensors, evidence, alpha=alpha, beta=beta)
    assert cohort.members == tuple(sensors)
    assert cohort.selection_cutoff == evidence['available_at'].max()
    rows = cohort.predict_all(bars)
    weights = []
    for sensor in sensors:
        observed = evidence.filter(pl.col('sensor_id') == sensor.state_id)
        correct = sum(prediction == outcome for prediction, outcome in observed.select('prediction', 'outcome').iter_rows())
        weights.append((alpha + correct) / (alpha + beta + observed.height))
    predictions = [sensor.predict_all(bars) for sensor in sensors]
    disagreements = 0
    for index, result in enumerate(rows):
        if result.datetime < cohort.selection_cutoff:
            assert result.reason == 'inside-training-window'
            assert result.prediction is result.probability is None
            continue
        votes = [batch[index].prediction for batch in predictions]
        disagreements += len(set(votes)) > 1
        expected = int(fsum(weight * vote for weight, vote in zip(weights, votes, strict=True)) > fsum(weights) / 2)
        assert result.prediction == expected
        assert result.probability is None and result.reason is None
    assert disagreements > 0
    reordered = bayesian_cohort(list(reversed(sensors)), evidence.reverse(), alpha=alpha, beta=beta)
    assert [row.prediction for row in reordered(bars)] == [row.prediction for row in rows]
    assert cohort.predict(bars) == rows[-1]
    assert cohort(bars) == rows
    assert [sensor.fitted_state for sensor in sensors] == [sensor.fitted_state for sensor in cohort.members]


def test_bayesian_priors_and_cutoff(recorded):
    bars, sensors, evidence = recorded
    empty = evidence.clear()
    cold = bayesian_cohort(sensors, empty)
    assert cold.selection_cutoff == max(sensor.selection_cutoff for sensor in sensors)
    batches = [sensor(bars) for sensor in sensors]
    for index, result in enumerate(cold(bars)):
        if result.reason is None:
            assert result.prediction == int(sum(batch[index].prediction for batch in batches) > len(sensors) / 2)
    cutoff = bars['datetime'][900]
    selected = bayesian_cohort(sensors, evidence, selection_cutoff=cutoff,
                              prepare=lambda history, sensor: history, state_id='recorded-bayesian')
    assert selected.selection_cutoff == cutoff and selected.state_id == 'recorded-bayesian'
    assert selected(bars)[899].reason == 'inside-training-window'
    assert selected(bars)[900].reason is None
    one = bayesian_cohort(sensors[:1], evidence.filter(pl.col('sensor_id') == sensors[0].state_id))
    assert one.predict(bars).prediction == sensors[0].predict(bars).prediction
    partial = bayesian_cohort(sensors, evidence.filter(pl.col('sensor_id') == sensors[0].state_id))
    assert partial.predict(bars).reason is None


@pytest.mark.parametrize('alpha,beta', [(1e-300, 1e300), (1e300, 1e-300), (5e-324, 1e300)])
def test_bayesian_extreme_priors_preserve_unanimous_recorded_votes(recorded, alpha, beta):
    bars, sensors, evidence = recorded
    empty = evidence.clear()
    expected = bayesian_cohort(sensors, empty)(bars)
    actual = bayesian_cohort(sensors, empty, alpha=alpha, beta=beta)(bars)
    assert actual == expected
    sensor = sensors[1]
    single = bayesian_cohort([sensor], empty, alpha=alpha, beta=beta)(bars)
    assert any(row.prediction == 1 for row in sensor(bars))
    assert [row.prediction for row in single] == [row.prediction for row in sensor(bars)]


@pytest.mark.parametrize('alpha,beta', [(0, 1), (-1, 1), (1, 0), (1, float('nan')),
                                        (float('inf'), 1), (True, 1), (1e308, 1e308)])
def test_bayesian_invalid_priors(recorded, alpha, beta):
    _, sensors, evidence = recorded
    with pytest.raises(ValueError, match='priors'):
        bayesian_cohort(sensors, evidence, alpha=alpha, beta=beta)


@pytest.mark.parametrize('change,message', [
    (lambda frame: frame.drop('source'), 'requires columns'),
    (lambda frame: frame.with_columns(pl.lit(None).alias('source')), 'null'),
    (lambda frame: frame.with_columns(pl.lit(' ').alias('source')), 'nonblank'),
    (lambda frame: frame.with_columns(pl.lit(1).alias('sensor_id')), 'nonblank'),
    (lambda frame: frame.with_columns((pl.col('sensor_id') + '-unknown').alias('sensor_id')), 'unknown Sensor'),
    (lambda frame: pl.concat([frame, frame.head(1)]), 'repeats'),
    (lambda frame: frame.with_columns(pl.when(pl.col('sensor_id') == frame['sensor_id'][0])
                                     .then(1 - pl.col('outcome')).otherwise(pl.col('outcome')).alias('outcome')), 'conflicting'),
    (lambda frame: frame.with_columns(pl.col('datetime').alias('available_at')), 'after'),
    (lambda frame: frame.with_columns(pl.col('available_at').dt.replace_time_zone(None)), 'UTC Datetime'),
    (lambda frame: frame.with_columns(pl.col('datetime').dt.epoch('ns')), 'UTC Datetime'),
    (lambda frame: frame.with_columns(pl.col('datetime').cast(pl.Datetime('ns', 'UTC')) + pl.duration(nanoseconds=1)), 'nanosecond'),
    (lambda frame: frame.with_columns(pl.lit(2).alias('outcome')), 'numeric binary'),
    (lambda frame: frame.with_columns(pl.lit(float('nan')).alias('prediction')), 'numeric binary'),
    (lambda frame: frame.with_columns(pl.col('prediction').cast(pl.Boolean)), 'numeric binary'),
    (lambda frame: frame.with_columns(pl.col('outcome').cast(pl.String)), 'numeric binary'),
    (lambda frame: frame.with_columns(pl.col('datetime') - pl.duration(days=30)), 'selection history'),
])
def test_bayesian_invalid_evidence(recorded, change, message):
    _, sensors, evidence = recorded
    with pytest.raises(ValueError, match=message):
        bayesian_cohort(sensors, change(evidence))


def test_bayesian_rejects_nonbinary_live_predictions(recorded):
    bars, sensors, evidence = recorded
    original = sensors[1]
    altered = Sensor.from_model(original.fitted_state, selection_cutoff=original.selection_cutoff,
                                feature_columns=['return_1', 'return_3'], state_id=original.state_id,
                                map_output=lambda output: [(float(value) / 2, None) for value in output],
                                output_id=original.output_id)
    with pytest.raises(ValueError, match='predictions must be binary'):
        bayesian_cohort([sensors[0], altered], evidence)(bars)
    with pytest.raises(ValueError, match='live Python Sensors'):
        bayesian_cohort([Sensor.__new__(Sensor)], evidence)
