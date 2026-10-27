from functools import partial
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from limen.log._experiment_backtest_results import _prepare_snapshot_backtest_input
from limen.log._snapshot_backtest_round import snapshot_backtest_round
from limen.sfd.reference_architecture import LogRegBinary
from limen.yaml import CompiledSFD

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def recorded_frames():
    bars = pl.read_parquet(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet').head(1200)
    config = {
        'metadata': {'name': 'snapshot_prediction_kind'},
        'sfd': {'manifest': {
            'type': 'ml',
            'data_source': {'method': 'limen.data.HistoricalData.get_spot_klines',
                            'params': {'kline_size': 3600}},
            'split_dates': {
                'train_start': '2024-01-01', 'train_end': '2024-01-23',
                'val_start': '2024-01-24', 'val_end': '2024-01-30',
                'test_start': '2024-01-31', 'test_end': '2024-02-06',
            },
            'indicators': [{'func': 'limen.indicators.window_return', 'params': {'period': 1}}],
            'features': [{'func': 'limen.features.lag_range',
                          'params': {'col': 'ret_1', 'start': 1, 'end': 5}}],
            'target': {'name': 'quantile_flag', 'class': 'limen.targets.QuantileBinaryTarget',
                       'fit_params': {'source_column': 'ret_1', 'quantile': 0.5},
                       'transform_params': {'shift': -1}},
            'scaler': {'class': 'limen.scalers.RobustScaler'},
            'strict_mode': True,
            'reference_architecture': 'limen.sfd.reference_architecture.logreg_binary',
        }, 'params': {'C': [1.0], 'max_iter': [1000]}},
    }
    data = CompiledSFD(config).manifest().prepare_data(bars, {'C': 1.0, 'max_iter': 1000})
    outputs = LogRegBinary().train(data, max_iter=1000).predict(data)
    probabilities = np.asarray(outputs['_probs'])
    assert ((probabilities > 0) & (probabilities < 1)).any()
    differences = (bars['close'] - bars['open']).to_numpy()
    assert (differences < 0).any() and (differences > 0).any()
    assert ((differences < 0) | (differences > 1)).any()
    assert (~np.isin(differences, [0, 1])).any()

    binary = pd.DataFrame({'predictions': outputs['_preds'], 'actuals': np.asarray(data['y_test']),
                           'model': 'LogRegBinary'})
    binary.index.name = 'recorded_row'
    binary = binary.iloc[::-1]
    observed = pd.read_csv(ROOT / 'tests/fixtures/historical_data_spot_2h.csv')
    labels = observed['no_of_trades']
    assert labels.nunique() > 2 and (labels >= 0).all()
    decisions = np.asarray(outputs['_preds'])[:len(observed)]
    frames = {
        'binary': binary,
        'probabilities': pd.DataFrame({'predictions': probabilities}),
        'directional': pd.DataFrame({'predictions': differences}),
        'nonnumeric_predictions': pd.DataFrame({'predictions': observed['datetime']}),
        'nonnumeric_actuals': pd.DataFrame({'predictions': decisions, 'actuals': observed['datetime']}),
        'multiclass_predictions': pd.DataFrame({'predictions': labels}),
        'multiclass_actuals': pd.DataFrame({'predictions': decisions, 'actuals': labels}),
    }
    return {name: pl.from_pandas(frame) for name, frame in frames.items()}


@pytest.mark.parametrize('prediction_kind', (None, 'binary', 'directional'))
@pytest.mark.parametrize('as_float', (False, True))
def test_recorded_binary_predictions_remain_unchanged(recorded_frames, prediction_kind, as_float):
    frame = recorded_frames['binary'].clone()
    if as_float:
        frame = frame.with_columns(pl.col('predictions').cast(pl.Float64))
    before = frame.clone()
    kwargs = {} if prediction_kind is None else {'prediction_kind': prediction_kind}
    result = _prepare_snapshot_backtest_input(frame, **kwargs)
    assert result is not frame
    assert_frame_equal(result, before)
    assert_frame_equal(frame, before)


@pytest.mark.parametrize('prediction_kind', (None, 'binary'))
@pytest.mark.parametrize('source', ('probabilities', 'directional'))
def test_default_and_binary_reject_recorded_nonbinary_predictions(recorded_frames, prediction_kind, source):
    kwargs = {} if prediction_kind is None else {'prediction_kind': prediction_kind}
    with pytest.raises(ValueError) as error:
        _prepare_snapshot_backtest_input(recorded_frames[source], **kwargs)
    for text in ('binary', "prediction_kind='directional'", 'threshold probabilities upstream'):
        assert text in str(error.value)


@pytest.mark.parametrize('source', ('probabilities', 'directional'))
def test_explicit_directional_preserves_sign_semantics(recorded_frames, source):
    frame = recorded_frames[source]
    before = frame.clone()
    result = _prepare_snapshot_backtest_input(frame, prediction_kind='directional')
    expected = before.clone()
    expected = before.with_columns((pl.col('predictions') > 0).cast(pl.Int64))
    assert_frame_equal(result, expected)
    assert_frame_equal(frame, before)


@pytest.mark.parametrize('prediction_kind', ('probability', None, 0))
@pytest.mark.parametrize('source', ('binary', 'nonnumeric_predictions'))
def test_unknown_prediction_kind_raises(recorded_frames, prediction_kind, source):
    with pytest.raises(ValueError) as error:
        _prepare_snapshot_backtest_input(recorded_frames[source], prediction_kind=prediction_kind)
    for text in ('prediction_kind', 'binary', 'directional'):
        assert text in str(error.value)


@pytest.mark.parametrize('prediction_kind', (None, 'binary', 'directional'))
@pytest.mark.parametrize(('source', 'message'), (
    ('nonnumeric_predictions', 'received non-numeric prediction values'),
    ('nonnumeric_actuals', 'received non-numeric actual values'),
    ('multiclass_predictions', 'does not support multiclass predictions'),
    ('multiclass_actuals', 'does not support multiclass actuals'),
))
def test_existing_invalid_inputs_keep_their_errors(recorded_frames, prediction_kind, source, message):
    kwargs = {} if prediction_kind is None else {'prediction_kind': prediction_kind}
    with pytest.raises(ValueError, match=message):
        _prepare_snapshot_backtest_input(recorded_frames[source], **kwargs)


@pytest.mark.parametrize('prediction_kind', (None, 'binary'))
@pytest.mark.parametrize('source', ('probabilities', 'directional'))
def test_rejected_nonbinary_inputs_do_not_reach_snapshot_accounting(recorded_frames, prediction_kind, source, monkeypatch):
    def accounting_was_entered(*args, **kwargs):
        pytest.fail('Nonbinary predictions reached snapshot accounting')

    monkeypatch.setattr(import_module('limen.log._snapshot_backtest_round'), 'backtest_snapshot', accounting_was_entered)
    log = SimpleNamespace(permutation_prediction_performance=lambda _: recorded_frames[source])
    kwargs = {} if prediction_kind is None else {'prediction_kind': prediction_kind}
    with pytest.raises(ValueError) as error:
        snapshot_backtest_round(log, 0, partial(_prepare_snapshot_backtest_input, **kwargs))
    for text in ('binary', "prediction_kind='directional'", 'threshold probabilities upstream'):
        assert text in str(error.value)
