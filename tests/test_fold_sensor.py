import copy
from datetime import timedelta
from unittest.mock import patch

import polars as pl
import pytest

from limen.cohort import Cohort
from limen.inference import Sensor
from limen.inference._python_sensor import decision_rows
from tests.fold_components import fold_campaigns


@pytest.mark.parametrize('kind', ['anchored', 'rolling', 'continuous', 'custom', 'formed', 'parallel', 'signed', 'reordered'])
@pytest.mark.parametrize('failure', ['none', 'feature', 'model'])
def test_guard_failure_precedence(fold_campaigns, kind, failure):
    item = fold_campaigns[kind]
    sensor = copy.deepcopy(item.sensor)
    source = item.bars.sort('datetime')
    # Shift actual recorded rows onto a later clock; prices remain recorded values.
    from tests.fold_components import FIXTURE
    history = pl.read_parquet(FIXTURE, n_rows=600)
    manifest = sensor._get_manifest()
    if failure == 'feature':
        context = patch.object(manifest, '_prepare_column_shims', side_effect=ValueError('feature failure'))
    elif failure == 'model':
        context = patch.object(sensor._model, 'predict', side_effect=ValueError('model failure'))
    else:
        context = patch.object(manifest, 'val_predict_guard', False)
    with context:
        rows = sensor.predict_all(history)
        assert rows
        assert sensor.predict(history) == rows[-1]
        for row in rows:
            assert row.datetime is not None
            if row.datetime < sensor.selection_cutoff:
                assert row.reason == 'inside-training-window'
                assert row.prediction is None and row.probability is None
            elif failure != 'none':
                assert row.reason == 'sensor-error'
        assert any(row.datetime >= sensor.selection_cutoff for row in rows)
        if kind == 'formed':
            assert len(rows) < len(history)
    # Original, pre-source and omitted native rows remain guarded, independent of flags.
    first = source.head(40)
    assert all(row.reason == 'inside-training-window' for row in sensor.predict_all(first))


@pytest.mark.parametrize('shape', ['duplicate', 'unsorted', 'null', 'naive', 'submicrosecond', 'finer', 'interval'])
def test_fold_sensor_clock(fold_campaigns, shape):
    sensor = fold_campaigns['anchored'].sensor
    frame = fold_campaigns['anchored'].bars.head(4)
    if shape == 'duplicate': frame = pl.concat([frame, frame.tail(1)])
    if shape == 'unsorted': frame = frame.reverse()
    if shape == 'null': frame = frame.with_columns(pl.lit(None, dtype=pl.Datetime('us', 'UTC')).alias('datetime'))
    if shape == 'naive': frame = frame.with_columns(pl.col('datetime').dt.replace_time_zone(None))
    if shape == 'submicrosecond': frame = frame.with_columns((pl.col('datetime').dt.epoch('ns') + 1).cast(pl.Datetime('ns', 'UTC')))
    if shape == 'finer': frame = frame.with_columns(pl.col('datetime') + pl.duration(minutes=30))
    if shape == 'interval': frame = frame.with_columns(pl.lit(1800).alias('base_interval'))
    with pytest.raises(ValueError):
        sensor.predict_all(frame)
    if shape not in ('finer', 'interval'):
        with pytest.raises(ValueError):
            decision_rows(frame)


@pytest.mark.parametrize('route', ['members', 'rounds', 'python', 'manual'])
def test_legacy_cohort_rejects_fold_binding(fold_campaigns, route):
    item = fold_campaigns['anchored']
    if route == 'manual':
        with pytest.raises(ValueError, match='validated Trainer'):
            Sensor(item.config, item.sensor._model, item.sensor._fitted_params, item.sensor.round_params)
    elif route == 'python':
        with pytest.raises(ValueError, match='fold-qualified'):
            Cohort.from_python([item.sensor], map_output=lambda value: value, mapped_output_id='binary', selection_cutoff=item.sensor.selection_cutoff)
    else:
        if route == 'rounds':
            with pytest.raises(ValueError, match='walk-forward rounds cannot be promoted'):
                Cohort(rounds=[(item.path, item.record['round_id'])])
        else:
            from limen.inference._fold_sensor import reject_legacy_members
            with pytest.raises(ValueError, match='fold-qualified'):
                reject_legacy_members([item.sensor])
