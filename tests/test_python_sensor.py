from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
import hashlib
from pathlib import Path
import runpy
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from limen import Cohort, Sensor
from limen.sfd.reference_architecture.logreg_binary import LogRegBinary


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/spot_15m_20250101_20250531.parquet'
FIXTURE_SHA = 'eb3392d7b4ccbd17a45ad9996e1c51d0f1b004c908a4a0264ff7efa959f1999a'
FITTING_CUTOFF = datetime(2025, 1, 11, tzinfo=timezone.utc)
SELECTION_CUTOFF = datetime(2025, 1, 16, tzinfo=timezone.utc)


@pytest.fixture(scope='module', autouse=True)
def forbid_network():
    with patch('requests.sessions.Session.request', side_effect=AssertionError('use recorded data')):
        yield


@pytest.fixture(scope='module')
def recorded():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA
    return pl.read_parquet(FIXTURE, n_rows=1920)


@pytest.fixture(scope='module')
def example(recorded, tmp_path_factory):
    document = (ROOT / 'docs/Universal-Experiment-Loop.md').read_text()
    marked = document.split('<!-- python-sensor-example -->', 1)[1]
    source = marked.split('```python\n', 1)[1].split('\n```', 1)[0]
    directory = tmp_path_factory.mktemp('python_sensor_example')
    script = ROOT / 'docs/examples/python-research/research.py'
    canonical = script.read_text().split('# region experiment\n', 1)[1].split('# endregion experiment', 1)[0].rstrip()
    assert source == canonical
    api = runpy.run_path(str(script))
    stem = directory / 'research'
    fitted_objects = []
    original_fit = Pipeline.fit

    def counted_fit(model, *args, **kwargs):
        fitted_objects.append(model)
        return original_fit(model, *args, **kwargs)

    with patch.object(Pipeline, 'fit', new=counted_fit):
        cohort, uel, selected, retained = api['run_research'](recorded, str(stem))
    return api, cohort, uel, selected, retained, stem, fitted_objects


def _metadata(recorded):
    return recorded[1440:1443].select('datetime').with_columns(
        (pl.col('datetime').dt.epoch('ns') + 900_000_000_000).alias('available_at_ns')
    )


def _sensor(metadata, mapped=None, **changes):
    pairs = [(1.25, None)] * metadata.height if mapped is None else mapped
    config = {
        'fitted_state': {}, 'params': {},
        'infer': lambda history, state, params: (metadata, pairs),
        'map_output': lambda output: output, 'state_id': 'recorded-state',
        'fitting_cutoff': FITTING_CUTOFF, 'selection_cutoff': SELECTION_CUTOFF,
    }
    config.update(changes)
    return Sensor.from_python(**config)


def _assert_predictions(actual, expected):
    assert [(p.datetime, p.reason, p.available_at_ns, p.trade_contract_digest) for p in actual] == [
        (p.datetime, p.reason, p.available_at_ns, p.trade_contract_digest) for p in expected
    ]
    assert [p.probability for p in actual] == [p.probability for p in expected]
    np.testing.assert_allclose(
        [p.prediction for p in actual], [p.prediction for p in expected],
        rtol=1e-12, atol=1e-12,
    )


def test_recorded_public_handoff_reproduces_selection_metric(example, recorded):
    api, cohort, uel, selected, retained, stem, fitted_objects = example
    assert isinstance(cohort, Cohort)
    assert cohort.origin == 'python'
    assert len(fitted_objects) == len(retained) == 4
    assert {id(model) for model in fitted_objects} == {id(model) for model in retained.values()}
    members = [(sensor.round_params, sensor.fitted_state) for sensor in cohort.members]
    assert uel.data.equals(recorded[:1440])
    assert uel.experiment_log.height == 4
    assert uel.models == []
    assert len(selected) == len(members) == 2
    assert [p['window'] for p in selected] == [4, 12]
    for sensor in cohort.members:
        assert isinstance(sensor, Sensor)
        assert sensor.fitted_state is retained[(sensor.round_params['window'], sensor.round_params['alpha'])]
        assert list(sensor.fitted_state.feature_names_in_) == ['momentum']
        assert sensor.output_id == 'next_bar_return'
    assert stem.with_suffix('.csv').is_file()
    assert Path(api['__file__']).resolve() == ROOT / 'docs/examples/python-research/research.py'
    assert set(stem.parent.iterdir()) == {stem.with_suffix('.csv')}
    close, opening = recorded['close'].to_numpy(), recorded['open'].to_numpy()
    member_scores = []
    for params, fitted in members:
        window = params['window']
        prepared = api['prep'](recorded[:1440], params)
        heldout = api['prepare_split'](recorded[1440:1920], params)
        assert [prepared['train'].height, prepared['validation'].height, heldout.height] == [947, 467, 467]
        for name, start, stop in (('train', 0, 960), ('validation', 960, 1440)):
            np.testing.assert_allclose(
                prepared[name]['next_return'].to_numpy(),
                close[start + 13:stop] / opening[start + 13:stop] - 1,
                rtol=1e-12, atol=1e-12,
            )
        # Independent source-row oracle: decisions 972..1438 trade rows 973..1439.
        inputs = pl.DataFrame({'momentum': close[972:1439] / close[972 - window:1439 - window] - 1})
        forecasts = fitted.predict(inputs)
        positions = np.sign(forecasts)
        next_bar_returns = close[973:1440] / opening[973:1440] - 1
        oracle = np.mean(positions * next_bar_returns - abs(positions) * 0.0002)
        rows = uel.experiment_log.filter(
            (pl.col('window') == window) & (pl.col('alpha') == params['alpha'])
        )
        logged = rows['validation_net_return'][0]
        evaluated = api['backtest'](prepared['validation'], forecasts)
        assert logged == pytest.approx(oracle, rel=1e-12, abs=1e-12)
        assert evaluated == pytest.approx(oracle, rel=1e-12, abs=1e-12)
        ranked = uel.experiment_log.filter(pl.col('window') == window).sort(
            ['validation_net_return', 'alpha'], descending=[True, False]
        )
        assert params['alpha'] == ranked['alpha'][0]
        heldout_input = pl.DataFrame({'momentum': close[1440:1920] / close[1440 - window:1920 - window] - 1})
        member_scores.append(fitted.predict(heldout_input))

    combined = cohort.predict_all(recorded[1428:1920])[12:]
    assert len(combined) == 480
    assert all(p.reason is None and p.probability is None for p in combined)
    assert [p.available_at_ns for p in combined] == (
        recorded[1440:1920]['datetime'].dt.epoch('ns') + 900_000_000_000
    ).to_list()
    expected = np.mean(member_scores, axis=0)
    np.testing.assert_allclose([p.prediction for p in combined], expected, rtol=1e-12, atol=1e-12)
    assert any(abs(member_scores[0] - member_scores[1]) > 1e-12)
    assert any(abs(expected - member_scores[0]) > 1e-12)
    assert any(abs(expected - member_scores[1]) > 1e-12)

    # The first selected member can predict; the second needs twelve prior bars.
    short_history = recorded[1432:1441]
    first_params, first_model = members[0]
    first_x = close[1440] / close[1440 - first_params['window']] - 1
    assert np.isfinite(first_model.predict(pl.DataFrame({'momentum': [first_x]}))[0])
    blocked = cohort.predict(short_history)
    assert blocked.datetime == SELECTION_CUTOFF
    assert blocked.reason == 'warm-up'
    assert blocked.prediction is blocked.probability is None


def test_recorded_prefixes_and_repeated_calls_never_fit(example, recorded):
    _, cohort, _, _, _, _, _ = example
    members = [(sensor.round_params, sensor.fitted_state) for sensor in cohort.members]
    history = recorded[1428:1920]
    snapshots = []
    with ExitStack() as stack:
        for _, fitted in members:
            scaler, ridge = fitted.named_steps['standardscaler'], fitted.named_steps['ridge']
            snapshots.append((scaler.mean_.copy(), scaler.scale_.copy(), ridge.coef_.copy(), ridge.intercept_))
            for component, method in ((fitted, 'fit'), (scaler, 'fit'), (scaler, 'fit_transform'), (ridge, 'fit')):
                stack.enter_context(patch.object(component, method, side_effect=AssertionError('inference cannot fit')))
        full = cohort.predict_all(history)
        for end in range(13, 493):
            prefix = cohort.predict_all(history[:end])
            _assert_predictions(prefix[12:], full[12:end])
        for end in (13, 253, 492):
            _assert_predictions([cohort.predict(history[:end])], [full[end - 1]])
    for (_, fitted), snapshot in zip(members, snapshots, strict=True):
        scaler, ridge = fitted.named_steps['standardscaler'], fitted.named_steps['ridge']
        for actual, expected in zip((scaler.mean_, scaler.scale_, ridge.coef_, ridge.intercept_), snapshot, strict=True):
            np.testing.assert_array_equal(actual, expected)


def test_arbitrary_history_state_params_and_identity(recorded):
    metadata = _metadata(recorded)
    history = {'completed': recorded[1440:1443], 'opaque': object()}
    state = {'offset': 1.0, 'opaque': np.array([1, 2])}
    nested = {'unit': 'basis-points'}
    params = {'scale': 2.0, 'nested': nested}

    def infer(supplied, fitted_state, supplied_params):
        assert supplied is history and fitted_state is state
        assert supplied_params['nested'] is nested
        with pytest.raises(TypeError):
            supplied_params['scale'] = 10
        return metadata, state['offset'] * supplied_params['scale']

    sensor = _sensor(metadata, fitted_state=state, params=params, infer=infer,
                     map_output=lambda output: [(output, None)] * metadata.height,
                     state_id=' caller-state ')
    params['scale'] = 99.0
    exposed = sensor.round_params
    exposed['scale'] = 100.0
    nested['unit'] = 'shared-nested-value'
    assert sensor.predict(history).prediction == 2.0
    state['offset'] = -2.0
    assert sensor.predict(history).prediction == -4.0
    assert sensor.round_params['nested'] is nested
    assert sensor.state_id == ' caller-state '
    assert sensor.origin == sensor.prediction_mode == 'python'
    assert sensor.permutation_id is sensor.manifest_id is sensor.trade_contract is None
    assert sensor.fitting_cutoff == FITTING_CUTOFF
    assert sensor.selection_cutoff == SELECTION_CUTOFF
    assert hash(sensor) == hash(sensor)
    other = _sensor(metadata, fitted_state=state)
    assert sensor != other and sensor not in [other]
    for name in ('state_id', 'fitting_cutoff', 'selection_cutoff', 'origin', 'prediction_mode', 'permutation_id', 'manifest_id'):
        with pytest.raises(AttributeError):
            setattr(sensor, name, None)


def test_selection_guard_and_declared_unavailability(recorded):
    metadata = recorded[1438:1443].select('datetime').with_columns(
        (pl.col('datetime').dt.epoch('ns') + 900_000_000_000).alias('available_at_ns'),
        pl.Series('reason', ['warm-up', None, None, 'warm-up', 'null-features']),
        pl.lit('ignored').alias('extra_column'),
    )
    sensor = _sensor(metadata, mapped=[(-2.0, 0.25)] * 5)
    predictions = sensor.predict_all(recorded)
    assert [p.reason for p in predictions] == [
        'inside-training-window', 'inside-training-window', None, 'warm-up', 'null-features'
    ]
    assert predictions[2].datetime == SELECTION_CUTOFF
    assert predictions[2].prediction == -2.0 and predictions[2].probability == 0.25
    for i, result in enumerate(predictions):
        assert result.available_at_ns == metadata['available_at_ns'][i]
        assert result.trade_contract_digest is None
        if i != 2:
            assert result.prediction is result.probability is None
    for unit in ('ms', 'us', 'ns'):
        converted = metadata.with_columns(pl.col('datetime').cast(pl.Datetime(unit, 'UTC')))
        assert _sensor(converted).predict_all(recorded)[2].datetime == SELECTION_CUTOFF
    nullable = metadata.select('datetime').with_columns(
        pl.lit(None).alias('reason'), pl.lit(None).alias('available_at_ns')
    )
    assert _sensor(nullable).predict(recorded).available_at_ns is None
    normalized = _sensor(metadata, fitting_cutoff=SELECTION_CUTOFF.astimezone(timezone(timedelta(hours=2))))
    assert normalized.fitting_cutoff == normalized.selection_cutoff == SELECTION_CUTOFF
    calls = []

    def empty_infer(history, state, params):
        calls.append(history)
        return metadata.head(0), []

    empty = _sensor(metadata, infer=empty_infer)
    assert empty.predict_all(recorded) == []
    result = empty.predict(recorded)
    assert result.reason == 'warm-up' and result.datetime is None
    assert result.prediction is result.probability is None
    assert len(calls) == 2


@pytest.mark.parametrize(('changes', 'error'), [
    ({'params': []}, TypeError), ({'params': {1: 'not-string'}}, TypeError),
    ({'state_id': None}, TypeError), ({'state_id': ''}, ValueError),
    ({'state_id': ' \t '}, ValueError), ({'infer': None}, TypeError),
    ({'map_output': None}, TypeError), ({'fitting_cutoff': '2025-01-11'}, TypeError),
    ({'selection_cutoff': None}, TypeError),
    ({'fitting_cutoff': datetime(2025, 1, 11)}, ValueError),
    ({'selection_cutoff': datetime(2025, 1, 16)}, ValueError),
    ({'fitting_cutoff': SELECTION_CUTOFF + timedelta(seconds=1)}, ValueError),
])
def test_invalid_construction(recorded, changes, error):
    with pytest.raises(error):
        _sensor(_metadata(recorded), **changes)


@pytest.mark.parametrize('fault', [
    'not-frame', 'missing-datetime', 'naive', 'non-utc', 'string-datetime',
    'null-datetime', 'duplicate', 'reversed', 'nanosecond-loss', 'unknown-reason',
    'sensor-error', 'inside-training-window', 'categorical-reason',
    'wrong-availability-type', 'early-availability', 'missing-output',
    'short-output', 'extra-output', 'string-output', 'generator-output', 'matrix-output',
])
def test_malformed_metadata_and_cardinality(recorded, fault):
    metadata = _metadata(recorded)
    mapped = [(1.0, None)] * metadata.height
    transforms = {
        'missing-datetime': lambda f: f.drop('datetime'),
        'naive': lambda f: f.with_columns(pl.col('datetime').dt.replace_time_zone(None)),
        'non-utc': lambda f: f.with_columns(pl.col('datetime').dt.convert_time_zone('Europe/Helsinki')),
        'string-datetime': lambda f: f.with_columns(pl.col('datetime').cast(pl.String)),
        'null-datetime': lambda f: f.with_columns(pl.lit(None, dtype=pl.Datetime('ms', 'UTC')).alias('datetime')),
        'duplicate': lambda f: f.with_columns(pl.col('datetime').first().alias('datetime')),
        'reversed': lambda f: f.reverse(),
        'nanosecond-loss': lambda f: f.with_columns((pl.col('datetime').dt.epoch('ns') + 1).cast(pl.Datetime('ns', 'UTC')).alias('datetime')),
        'unknown-reason': lambda f: f.with_columns(pl.lit('unknown').alias('reason')),
        'sensor-error': lambda f: f.with_columns(pl.lit('sensor-error').alias('reason')),
        'inside-training-window': lambda f: f.with_columns(pl.lit('inside-training-window').alias('reason')),
        'categorical-reason': lambda f: f.with_columns(pl.lit('warm-up').cast(pl.Categorical).alias('reason')),
        'wrong-availability-type': lambda f: f.with_columns(pl.lit(1, dtype=pl.Int32).alias('available_at_ns')),
        'early-availability': lambda f: f.with_columns((pl.col('datetime').dt.epoch('ns') - 1).alias('available_at_ns')),
    }
    if fault in transforms:
        metadata = transforms[fault](metadata)
    elif fault == 'not-frame':
        metadata = None
    elif fault == 'missing-output':
        mapped = None
    elif fault == 'short-output':
        mapped = mapped[:-1]
    elif fault == 'extra-output':
        mapped = [*mapped, mapped[-1]]
    elif fault == 'string-output':
        mapped = 'invalid'
    elif fault == 'generator-output':
        mapped = iter(mapped)
    elif fault == 'matrix-output':
        mapped = np.ones((3, 2))
    sensor = _sensor(_metadata(recorded), infer=lambda history, state, params: (metadata, mapped))
    for method in (sensor.predict, sensor.predict_all):
        with pytest.raises(ValueError):
            method(recorded)


@pytest.mark.parametrize('invalid', [
    (None, None), (True, None), (np.bool_(True), None), ('up', None),
    (complex(1, 2), None), (float('nan'), None), (float('inf'), None),
    (float('-inf'), None), (np.array(2.0), None), (np.array([1.0, 2.0]), None),
    (Decimal('1.25'), None), (1.0, True), (1.0, float('nan')),
    (1.0, float('inf')), (1.0, -0.01), (1.0, 1.01),
    1.0, (), (1.0,), (1.0, None, None), 'invalid-pair',
])
def test_invalid_aligned_scalars(recorded, invalid):
    metadata = _metadata(recorded)
    sensor = _sensor(metadata, mapped=[(-2.0, None), (3.5, 1.0), invalid])
    results = sensor.predict_all(recorded)
    assert results[0].prediction == -2.0 and results[0].reason is None
    assert results[1].prediction == 3.5 and results[1].probability == 1.0
    assert results[2].reason == 'sensor-error'
    assert results[2].prediction is results[2].probability is None
    assert results[2].datetime == metadata['datetime'][-1]
    assert results[2].available_at_ns == metadata['available_at_ns'][-1]
    assert sensor.predict(recorded) == results[-1]
    accepted = _sensor(metadata, mapped=[(np.int64(-2), None), (np.float32(3.5), 0.0), (Fraction(5, 2), 1.0)])
    valid = accepted.predict_all(recorded)
    assert [p.prediction for p in valid] == [-2, 3.5, 2.5]
    assert type(valid[0].prediction) is int
    assert all(type(p.prediction) is float for p in valid[1:])
    assert all(p.reason is None for p in valid)


@pytest.mark.parametrize('stage', ['infer', 'adapter'])
def test_inference_and_adapter_failures(recorded, stage):
    metadata = _metadata(recorded)

    def fail(*args):
        raise RuntimeError('caller failure')

    changes = {'infer' if stage == 'infer' else 'map_output': fail}
    sensor = _sensor(metadata, **changes)
    for method in (sensor.predict, sensor.predict_all):
        with pytest.raises(RuntimeError, match=r'^caller failure$'):
            method(recorded)


def test_python_factory_preserves_compiled_construction(recorded):
    for yaml_reference, expected_error in ((None, TypeError), ({}, KeyError), ({'garbage': 1}, KeyError)):
        compiled = Sensor(yaml_reference=yaml_reference, model=LogRegBinary(), fitted_params={}, round_params={})
        for method in (compiled.predict, compiled.predict_all):
            with pytest.raises(expected_error):
                method(recorded[1440:1443])
    sensor = _sensor(_metadata(recorded))
    with patch('limen.yaml.compiler.CompiledSFD.manifest', side_effect=AssertionError('Python must not compile')):
        assert sensor.predict({'arbitrary': recorded}).reason is None
        assert len(sensor({'arbitrary': recorded})) == 3
