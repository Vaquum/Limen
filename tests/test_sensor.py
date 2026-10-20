from __future__ import annotations

import copy
from dataclasses import replace
from datetime import date
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from limen.backtest.trade_contract import contract_digest
from limen.experiment import MLManifest
from limen.experiment.manifest_core import ProductConfig
from limen.experiment.manifest_core import PCACompressionConfig
from limen.experiment.manifest_core import _apply_sensor_pca
from limen.experiment.manifest_core import _count_leading_nulls
from limen.inference.sensor import BarPrediction
from limen.inference.sensor import Sensor
from limen.inference.sensor import _extract_scalar
from limen.scalers.robust_scaler import RobustScaler


def _make_klines(n: int = 20, start_year: int = 2026) -> pl.DataFrame:
    idx = np.arange(n, dtype=float)
    close = 100.0 + idx * 0.5
    return pl.DataFrame({
        'datetime': pl.datetime_range(
            start=pl.datetime(start_year, 1, 1, 0, 0, 0),
            end=pl.datetime(start_year, 1, n + 1, 0, 0, 0),
            interval='1d',
            eager=True,
        )[:n],
        'open': close - 0.2,
        'high': close + 0.5,
        'low': close - 0.5,
        'close': close,
        'volume': 1000.0 + idx * 5.0,
    })


def _make_basic_manifest() -> MLManifest:
    return (
        MLManifest()
        .set_split_config(3, 1, 1)
    )


def _make_lag_manifest(lag: int = 3) -> tuple[MLManifest, int]:
    manifest = (
        MLManifest()
        .set_split_config(3, 1, 1)
        .add_indicator(
            lambda df, n=lag: df.with_columns(pl.col('close').shift(n).alias(f'lag_{n}'))
        )
    )
    return manifest, lag


def _make_stub_model(pred: int = 1, prob: float = 0.7) -> MagicMock:
    model = MagicMock()
    def _predict(data_dict):
        n = len(data_dict['x_test'])
        return {'_preds': np.array([pred] * n), '_probs': np.array([prob] * n)}
    model.predict.side_effect = _predict
    model.deterministic = True
    return model


def _make_sensor(
        manifest: Any,
        model: Any = None,
        fitted_params: dict | None = None,
        round_params: dict | None = None,
) -> Sensor:
    sensor = object.__new__(Sensor)
    sensor._yaml_reference = {}
    sensor._model = model or _make_stub_model()
    sensor._fitted_params = dict(fitted_params or {})
    sensor._round_params = dict(round_params or {'bar_type': 'base'})
    sensor._manifest = manifest
    sensor.permutation_id = None
    return sensor


def _make_pca_manifest_stub(component_prefix: str = 'pc') -> Any:
    manifest = object.__new__(MLManifest)
    manifest.pca_compression_config = PCACompressionConfig(
        enabled_param='use_pca',
        n_components_param='n_pca',
        scaler_param_name='_scaler',
        component_prefix=component_prefix,
    )
    manifest.target_column = None
    return manifest


def test_count_leading_nulls_no_nulls() -> None:
    data = pl.DataFrame({'datetime': [1, 2, 3], 'f': [1.0, 2.0, 3.0]})
    assert _count_leading_nulls(data) == 0


def test_count_leading_nulls_leading_nulls() -> None:
    data = pl.DataFrame({'f': [None, None, 1.0, 2.0, 3.0]})
    assert _count_leading_nulls(data) == 2


def test_count_leading_nulls_all_null() -> None:
    data = pl.DataFrame({'f': [None, None, None]})
    assert _count_leading_nulls(data) == 3


def test_count_leading_nulls_gap_in_middle() -> None:
    data = pl.DataFrame({'f': [1.0, None, 3.0]})
    assert _count_leading_nulls(data) == 0


def test_count_leading_nulls_datetime_excluded() -> None:
    data = pl.DataFrame({'datetime': [None, None], 'f': [1.0, 2.0]})
    assert _count_leading_nulls(data) == 0


def test_count_leading_nulls_any_null_col_triggers_row() -> None:
    data = pl.DataFrame({'f1': [None, 1.0, 2.0], 'f2': [0.0, 2.0, 3.0]})
    assert _count_leading_nulls(data) == 1


def test_count_leading_nulls_empty_features() -> None:
    data = pl.DataFrame({'datetime': [1, 2, 3]})
    assert _count_leading_nulls(data) == 0


def test_sensor_input_prep_preserves_row_count() -> None:
    manifest = _make_basic_manifest()
    raw = _make_klines(20)
    data, _ = manifest.sensor_input_prep(raw, {}, {'bar_type': 'base'})
    assert len(data) == len(raw)


def test_sensor_input_prep_indicator_lookback_matches_leading_nulls() -> None:
    manifest, lag = _make_lag_manifest(lag=4)
    raw = _make_klines(20)
    data, indicator_lookback = manifest.sensor_input_prep(raw, {}, {'bar_type': 'base'})
    assert indicator_lookback == lag
    assert len(data) == len(raw)


def test_sensor_input_prep_no_leading_nulls_returns_zero_lookback() -> None:
    manifest = _make_basic_manifest()
    raw = _make_klines(10)
    _, indicator_lookback = manifest.sensor_input_prep(raw, {}, {'bar_type': 'base'})
    assert indicator_lookback == 0


def test_sensor_input_prep_no_drop_nulls() -> None:
    manifest, lag = _make_lag_manifest(lag=3)
    raw = _make_klines(15)
    data, lookback = manifest.sensor_input_prep(raw, {}, {'bar_type': 'base'})
    assert len(data) == len(raw)
    assert lookback == 3
    assert data[f'lag_{lag}'][:lag].null_count() == lag


def test_sensor_input_prep_drops_ablated_features() -> None:
    manifest = (
        MLManifest()
        .set_split_config(3, 1, 1)
        .add_indicator(lambda df: df.with_columns(pl.col('close').alias('feat_a')))
        .add_indicator(lambda df: df.with_columns(pl.col('close').alias('feat_b')))
        .set_feature_ablation()
    )
    raw = _make_klines(20)
    round_params = {
        'bar_type': 'base',
        'feature_drop_count': 1,
        'feature_drop_seed': 42,
        '_dropped_features': ['feat_a'],
    }
    data, _ = manifest.sensor_input_prep(raw, {}, round_params)
    assert 'feat_a' not in data.columns
    assert 'feat_b' in data.columns


def test_sensor_input_prep_applies_fitted_scaler() -> None:
    manifest = (
        MLManifest()
        .set_split_config(3, 1, 1)
        .set_scaler(RobustScaler)
    )
    raw = _make_klines(20)
    train = raw.select([c for c in raw.columns if c != 'datetime'])
    fitted_params = {'_scaler': RobustScaler(train)}

    data, _ = manifest.sensor_input_prep(raw, fitted_params, {'bar_type': 'base'})

    assert len(data) == len(raw)
    assert data['close'].to_list() != raw['close'].to_list()


def test_sensor_input_prep_scaler_leaves_null_rows_null() -> None:
    manifest = (
        MLManifest()
        .set_split_config(3, 1, 1)
        .add_indicator(lambda df: df.with_columns(pl.col('close').shift(2).alias('lag')))
        .set_scaler(RobustScaler)
    )
    raw = _make_klines(20)
    train = raw.drop_nulls().select([c for c in raw.columns if c != 'datetime'])
    fitted_params = {'_scaler': RobustScaler(train)}

    data, lookback = manifest.sensor_input_prep(raw, fitted_params, {'bar_type': 'base'})

    assert lookback == 2
    assert len(data) == len(raw)
    assert data['lag'][:2].null_count() == 2


def test_apply_sensor_pca_transforms_valid_rows() -> None:
    from sklearn.decomposition import PCA

    n = 10
    data = pl.DataFrame({
        'datetime': list(range(n)),
        'f1': np.random.default_rng(0).standard_normal(n).tolist(),
        'f2': np.random.default_rng(1).standard_normal(n).tolist(),
    })
    pca = PCA(n_components=1, svd_solver='full')
    pca.fit(data.select(['f1', 'f2']).to_numpy())
    manifest = _make_pca_manifest_stub()
    all_fitted_params = {
        '_pca': pca,
        '_pca_input_feature_names': ['f1', 'f2'],
        '_pca_feature_names': ['pc0'],
    }

    out = _apply_sensor_pca(manifest, data, {'use_pca': True, 'n_pca': 1}, all_fitted_params)

    assert 'pc0' in out.columns
    assert len(out) == n
    assert out['pc0'].null_count() == 0


def test_apply_sensor_pca_preserves_null_rows() -> None:
    from sklearn.decomposition import PCA

    n = 10
    f1_vals = [None, None, *np.arange(8, dtype=float).tolist()]
    f2_vals = [None, None, *np.arange(8, 16, dtype=float).tolist()]
    data = pl.DataFrame({
        'datetime': list(range(n)),
        'f1': pl.Series(f1_vals, dtype=pl.Float64),
        'f2': pl.Series(f2_vals, dtype=pl.Float64),
    })
    pca = PCA(n_components=1, svd_solver='full')
    pca.fit(data.drop_nulls().select(['f1', 'f2']).to_numpy())
    manifest = _make_pca_manifest_stub()
    all_fitted_params = {
        '_pca': pca,
        '_pca_input_feature_names': ['f1', 'f2'],
        '_pca_feature_names': ['pc0'],
    }

    out = _apply_sensor_pca(manifest, data, {'use_pca': True, 'n_pca': 1}, all_fitted_params)

    assert len(out) == n
    assert out['pc0'][:2].null_count() == 2
    assert out['pc0'][2:].null_count() == 0


def test_apply_sensor_pca_skipped_when_disabled() -> None:
    manifest = _make_pca_manifest_stub()
    data = pl.DataFrame({'datetime': [1, 2], 'f1': [1.0, 2.0]})
    out = _apply_sensor_pca(manifest, data, {'use_pca': False}, {})
    assert 'f1' in out.columns
    assert 'pc0' not in out.columns


def test_apply_sensor_pca_skipped_when_no_config() -> None:
    manifest = object.__new__(MLManifest)
    manifest.pca_compression_config = None
    manifest.target_column = None
    data = pl.DataFrame({'f1': [1.0, 2.0]})
    out = _apply_sensor_pca(manifest, data, {}, {})
    assert out is data


def test_predict_returns_warmup_if_last_bar_is_warmup() -> None:
    manifest, _lag = _make_lag_manifest(lag=5)
    raw = _make_klines(5)
    sensor = _make_sensor(manifest)
    result = sensor.predict(raw)
    assert result.reason == 'warm-up'
    assert result.prediction is None


def test_predict_returns_warmup_if_insufficient_valid_rows() -> None:
    manifest, _ = _make_lag_manifest(lag=3)
    # decoder_lookback=1, only 3 rows → valid_rows=0 < 1
    raw = _make_klines(3)
    sensor = _make_sensor(manifest)
    result = sensor.predict(raw)
    assert result.reason == 'warm-up'
    assert result.prediction is None


def test_predict_raises_not_implemented_for_decoder_lookback_gt_1() -> None:
    manifest, _ = _make_lag_manifest(lag=3)
    manifest.decoder_lookback = 2
    raw = _make_klines(10)
    sensor = _make_sensor(manifest)
    try:
        sensor.predict(raw)
        assert False, 'Expected NotImplementedError'
    except NotImplementedError as e:
        assert 'decoder_lookback' in str(e)


def test_predict_returns_inside_training_window_reason() -> None:
    manifest = _make_basic_manifest()
    manifest.split_dates = (
        date(2026, 1, 1), date(2026, 1, 10),
        date(2026, 1, 10), date(2026, 1, 15),
        date(2026, 1, 15), date(2026, 1, 20),
    )
    raw = _make_klines(5, start_year=2026)
    sensor = _make_sensor(manifest)
    result = sensor.predict(raw)
    assert result.reason == 'inside-training-window'
    assert result.prediction is None


def test_predict_allows_context_bars_inside_window_when_last_bar_is_valid() -> None:
    manifest, _lag = _make_lag_manifest(lag=2)
    manifest.split_dates = (
        date(2026, 1, 1), date(2026, 1, 10),
        date(2026, 1, 10), date(2026, 1, 15),
        date(2026, 1, 15), date(2026, 1, 20),
    )
    # 19 context bars (inside window) + 1 post-test_end bar as the last bar
    raw = _make_klines(20, start_year=2026)
    sensor = _make_sensor(manifest, round_params={'bar_type': 'base'})
    result = sensor.predict(raw)
    assert isinstance(result, BarPrediction)


def test_predict_happy_path_returns_bar_prediction() -> None:
    manifest, _lag = _make_lag_manifest(lag=2)
    manifest.split_dates = None
    raw = _make_klines(10)
    sensor = _make_sensor(manifest, round_params={'bar_type': 'base'})
    result = sensor.predict(raw)
    assert isinstance(result, BarPrediction)
    assert result.reason is None
    assert result.prediction is not None


def test_predict_all_output_length_equals_input() -> None:
    manifest, _lag = _make_lag_manifest(lag=2)
    manifest.split_dates = None
    raw = _make_klines(10)
    sensor = _make_sensor(manifest, round_params={'bar_type': 'base'})
    results = sensor.predict_all(raw)
    assert len(results) == len(raw)


def test_predict_all_warmup_bars_have_reason_warmup() -> None:
    manifest, lag = _make_lag_manifest(lag=3)
    manifest.split_dates = None
    raw = _make_klines(10)
    sensor = _make_sensor(manifest, round_params={'bar_type': 'base'})
    results = sensor.predict_all(raw)
    for i in range(lag):
        assert results[i].reason == 'warm-up'
        assert results[i].prediction is None


def test_predict_all_valid_bars_have_predictions() -> None:
    manifest, lag = _make_lag_manifest(lag=2)
    manifest.split_dates = None
    raw = _make_klines(10)
    sensor = _make_sensor(manifest, round_params={'bar_type': 'base'})
    results = sensor.predict_all(raw)
    for i in range(lag, len(results)):
        assert results[i].reason is None
        assert results[i].prediction is not None


def test_predict_all_inside_window_bars_flagged() -> None:
    manifest = _make_basic_manifest()
    manifest.split_dates = (
        date(2026, 1, 1), date(2026, 1, 5),
        date(2026, 1, 5), date(2026, 1, 8),
        date(2026, 1, 8), date(2026, 1, 12),
    )
    raw = _make_klines(20, start_year=2026)
    sensor = _make_sensor(manifest, round_params={'bar_type': 'base'})
    results = sensor.predict_all(raw)
    inside = [r for r in results if r.reason == 'inside-training-window']
    assert len(inside) > 0
    for r in inside:
        assert r.prediction is None


def test_predict_all_midstream_null_row_has_reason_null_features() -> None:
    manifest, _lag = _make_lag_manifest(lag=2)
    manifest.split_dates = None
    raw = _make_klines(10)
    # inject a null into a non-leading row (row 5, well past warm-up)
    raw = raw.with_columns(
        pl.when(pl.int_range(pl.len()) == 5)
        .then(None)
        .otherwise(pl.col('close'))
        .alias('close')
    )
    sensor = _make_sensor(manifest, round_params={'bar_type': 'base'})
    results = sensor.predict_all(raw)
    assert results[5].reason == 'null-features'
    assert results[5].prediction is None
    assert results[5].probability is None
    # other post-warmup bars still produce predictions
    assert results[6].reason is None
    assert results[6].prediction is not None


def test_predict_all_raises_not_implemented_for_decoder_lookback_gt1() -> None:
    manifest = _make_basic_manifest()
    manifest.decoder_lookback = 2
    manifest.split_dates = None
    raw = _make_klines(10)
    sensor = _make_sensor(manifest, round_params={'bar_type': 'base'})
    try:
        sensor.predict_all(raw)
        assert False, 'Expected NotImplementedError'
    except NotImplementedError as e:
        assert 'decoder_lookback' in str(e)


def test_extract_scalar_returns_none_for_none() -> None:
    assert _extract_scalar(None) is None


def test_extract_scalar_returns_python_int() -> None:
    assert _extract_scalar(1) == 1
    assert isinstance(_extract_scalar(1), int)


def test_extract_scalar_returns_python_float() -> None:
    assert _extract_scalar(0.7) == 0.7
    assert isinstance(_extract_scalar(0.7), float)


@pytest.mark.parametrize('value', (True, False))
def test_extract_scalar_preserves_bool(value) -> None:
    assert _extract_scalar(value) is value
    assert _extract_scalar(np.bool_(value)) is value
    assert _extract_scalar([value]) is value
    assert _extract_scalar(np.array([value])) is value


def test_extract_scalar_from_numpy_array() -> None:
    arr = np.array([0.5, 0.3])
    result = _extract_scalar(arr)
    assert result == 0.5


def test_extract_scalar_from_zero_dim_numpy() -> None:
    arr = np.float64(0.42)
    result = _extract_scalar(arr)
    assert result == 0.42


def test_extract_scalar_from_empty_array_returns_none() -> None:
    assert _extract_scalar(np.array([])) is None


_TRAIN_START = date(2024, 1, 1)
_TRAIN_MID = date(2024, 1, 5)
_VAL_START = date(2024, 1, 10)
_VAL_LAST = date(2024, 1, 14)
_TEST_START = date(2024, 1, 15)
_TEST_LAST = date(2024, 1, 19)
_TEST_END = date(2024, 1, 20)
_PRE_TRAIN = date(2023, 12, 31)


def _guard_manifest(val_predict_guard: bool = True,
                    test_predict_guard: bool = True) -> MLManifest:
    return MLManifest().set_split_dates(
        _TRAIN_START, _VAL_START,
        _VAL_START, _TEST_START,
        _TEST_START, _TEST_END,
        val_predict_guard=val_predict_guard,
        test_predict_guard=test_predict_guard,
    )


def _guard_mask(manifest: MLManifest, days: list[date]) -> list[bool]:
    data = pl.DataFrame({'datetime': [datetime(d.year, d.month, d.day) for d in days]})
    sensor = _make_sensor(manifest)
    mask = sensor._inside_training_window_mask(data, manifest)
    last_bar = [
        sensor._last_bar_inside_training_window(datetime(d.year, d.month, d.day), manifest)
        for d in days
    ]
    assert mask == last_bar
    return mask


def test_guard_masks_all_three_windows_by_default() -> None:
    manifest = _guard_manifest()
    days = [_PRE_TRAIN, _TRAIN_START, _TRAIN_MID, _VAL_START, _VAL_LAST,
            _TEST_START, _TEST_LAST, _TEST_END]
    assert _guard_mask(manifest, days) == [
        False, True, True, True, True, True, True, False,
    ]


def test_test_predict_guard_false_serves_test_window() -> None:
    manifest = _guard_manifest(test_predict_guard=False)
    days = [_TRAIN_START, _TRAIN_MID, _VAL_START, _VAL_LAST,
            _TEST_START, _TEST_LAST, _TEST_END]
    assert _guard_mask(manifest, days) == [
        True, True, True, True, False, False, False,
    ]


def test_val_predict_guard_false_serves_val_only() -> None:
    manifest = _guard_manifest(val_predict_guard=False)
    days = [_TRAIN_START, _TRAIN_MID, _VAL_START, _VAL_LAST,
            _TEST_START, _TEST_LAST]
    assert _guard_mask(manifest, days) == [
        True, True, False, False, True, True,
    ]


def test_set_split_dates_stores_guard_flags() -> None:
    default = _guard_manifest()
    assert default.val_predict_guard is True
    assert default.test_predict_guard is True

    configured = _guard_manifest(val_predict_guard=False, test_predict_guard=False)
    assert configured.val_predict_guard is False
    assert configured.test_predict_guard is False

    for bad in ('true', 1, None):
        try:
            MLManifest().set_split_dates(
                _TRAIN_START, _VAL_START,
                _VAL_START, _TEST_START,
                _TEST_START, _TEST_END,
                val_predict_guard=bad,
            )
            assert False, 'Expected TypeError for non-bool val_predict_guard'
        except TypeError as e:
            assert 'val_predict_guard' in str(e)

    try:
        MLManifest().set_split_dates(
            _TRAIN_START, _VAL_START,
            _VAL_START, _TEST_START,
            _TEST_START, _TEST_END,
            test_predict_guard='no',
        )
        assert False, 'Expected TypeError for non-bool test_predict_guard'
    except TypeError as e:
        assert 'test_predict_guard' in str(e)


def _recorded_compiled_source(klines_size=900):
    assert klines_size == 900
    path = Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet'
    assert sha256(path.read_bytes()).hexdigest() == 'eb3392d7b4ccbd17a45ad9996e1c51d0f1b004c908a4a0264ff7efa959f1999a'
    return pl.read_parquet(path).head(500)


@pytest.fixture(scope='module')
def compiled_recording():
    raw = _recorded_compiled_source()
    manifest, _ = _make_lag_manifest(lag=3)
    manifest.set_data_source(_recorded_compiled_source, params={'klines_size': 900})
    manifest.set_backtest_config(
        prediction_mode='binary',
        product=ProductConfig('cash_spot', 'BTCUSDT', 'BTC', 'USDT', 1e-9, 0),
    )
    params = {'bar_type': 'base'}
    fitted = manifest.prepare_data(raw, params)
    manifest.set_split_dates(
        date(2025, 1, 2), date(2025, 1, 3),
        date(2025, 1, 3), date(2025, 1, 4),
        date(2025, 1, 4), date(2025, 1, 5),
    )
    manifest.add_indicator(lambda df: df.with_columns(
        pl.when(pl.col('datetime') == raw['datetime'][388])
        .then(None).otherwise(pl.col('lag_3')).alias('lag_3'),
    ))
    prepared, lookback = manifest.sensor_input_prep(raw, fitted['_fitted_params'], params)
    availability = raw['datetime'].dt.epoch('ns') + 900_000_000_000
    assert lookback == 3
    assert prepared['datetime'].equals(raw['datetime'])
    assert prepared['__trade_available_at_ns__'].equals(availability, check_names=False)
    digest = contract_digest(fitted['trade_contract'])
    assert digest == fitted['trade_contract_digest']
    expected = []
    for i, dt in enumerate(raw['datetime']):
        if date(2025, 1, 2) <= dt.date() < date(2025, 1, 5):
            reason = 'inside-training-window'
        elif i < 3:
            reason = 'warm-up'
        elif i == 388:
            reason = 'null-features'
        else:
            expected.append(BarPrediction(dt, -0.25, 0.5, None, int(availability[i]), digest))
            continue
        expected.append(BarPrediction(dt, None, None, reason))
    assert all(row.datetime is not None and row.available_at_ns is not None
               and row.trade_contract_digest for row in expected if row.reason is None)
    return raw, manifest, fitted, expected


@pytest.fixture
def compiled_sensor(compiled_recording):
    raw, manifest, fitted, expected = compiled_recording
    sensor = _make_sensor(copy.deepcopy(manifest), fitted_params=fitted['_fitted_params'])
    sensor._trade_contract = copy.deepcopy(fitted['trade_contract'])
    sensor._trade_digest = contract_digest(sensor._trade_contract)
    sensor._model.predict.side_effect = None
    return sensor, raw, expected


def _compiled_results(sensor, raw, single_output, batch_output):
    sensor._model.predict.return_value = batch_output
    results = sensor.predict_all(raw)
    sensor._model.predict.return_value = single_output
    return sensor.predict(raw), results


@pytest.mark.parametrize('array_output', (False, True))
@pytest.mark.parametrize('prediction,probability', [
    pair
    for scalar in (float, np.float64)
    for value in ('nan', 'inf', '-inf')
    for pair in ((scalar(value), 0.5), (-0.25, scalar(value)))
] + [
    (-0.25, scalar(value))
    for scalar in (float, np.float64)
    for value in (-0.1, 1.1)
] + [
    pair
    for scalar in (bool, np.bool_)
    for value in (True, False)
    for pair in ((scalar(value), 0.5), (-0.25, scalar(value)))
] + [(None, 0.5)])
def test_compiled_sensor_invalid_output_pairs(compiled_sensor, prediction, probability, array_output):
    sensor, raw, expected = compiled_sensor
    count = sum(row.reason is None for row in expected)
    pair = {'_preds': prediction, '_probs': probability}
    single = {key: np.array([value]) for key, value in pair.items()} if array_output else pair
    batch = {key: np.array([value] * count) if array_output else [value] * count
             for key, value in pair.items()}
    last, results = _compiled_results(sensor, raw, single, batch)
    rejected = [replace(row, prediction=None, probability=None, reason='sensor-error')
                if row.reason is None else row for row in expected]
    assert results == rejected
    assert last == rejected[-1]


@pytest.mark.parametrize('array_output', (False, True))
@pytest.mark.parametrize('prediction,probability,optional', [
    (-0.25, None, 'omitted'),
    (-0.25, None, 'none'),
    (-0.25, None, 'aligned'),
    (0, 0, 'aligned'),
    (1.25, 1, 'aligned'),
    (np.float64(-0.75), np.float64(0.2), 'aligned'),
    (np.int64(-1), np.int64(0), 'aligned'),
    (np.float32(0), np.float32(1), 'aligned'),
    (10 ** 40, None, 'aligned'),
    (-10 ** 40, None, 'aligned'),
])
def test_compiled_sensor_valid_output_pairs(compiled_sensor, prediction, probability, optional, array_output):
    sensor, raw, expected = compiled_sensor
    count = sum(row.reason is None for row in expected)
    pair = {'_preds': prediction, '_probs': probability}
    single = {key: np.array([value]) for key, value in pair.items()} if array_output else pair
    batch = {key: np.array([value] * count) if array_output else [value] * count
             for key, value in pair.items()}
    if optional == 'omitted':
        del single['_probs']
        del batch['_probs']
    elif optional == 'none':
        single['_probs'] = batch['_probs'] = None
    last, results = _compiled_results(sensor, raw, single, batch)
    accepted = [replace(row, prediction=prediction, probability=probability)
                if row.reason is None else row for row in expected]
    assert results == accepted
    assert last == accepted[-1]


@pytest.mark.parametrize('omitted', (False, True))
def test_compiled_sensor_missing_prediction_outputs(compiled_sensor, omitted):
    sensor, raw, expected = compiled_sensor
    single = {'_probs': 0.5}
    batch = {'_probs': [0.5] * sum(row.reason is None for row in expected)}
    if not omitted:
        single['_preds'] = batch['_preds'] = None
    last, results = _compiled_results(sensor, raw, single, batch)
    rejected = [replace(row, prediction=None, probability=None, reason='sensor-error')
                if row.reason is None else row for row in expected]
    assert results == rejected
    assert last == rejected[-1]


@pytest.mark.parametrize('bad_pair', [
    (None, 0.5), (float('nan'), 0.5), (-0.25, float('inf')),
    (False, 0.5), (-0.25, True), (-0.25, 1.1),
])
def test_compiled_sensor_mixed_output_alignment(compiled_sensor, bad_pair):
    sensor, raw, expected = compiled_sensor
    valid = [i for i, row in enumerate(expected) if row.reason is None]
    assert valid[94] == 385
    preds = [j - 0.5 for j in range(len(valid))]
    probs = [j % 2 for j in range(len(valid))]
    aligned = list(expected)
    for j, i in enumerate(valid):
        aligned[i] = replace(expected[i], prediction=preds[j], probability=probs[j])
    preds[94], probs[94] = bad_pair
    aligned[385] = replace(expected[385], prediction=None, probability=None, reason='sensor-error')
    last, results = _compiled_results(
        sensor, raw, {'_preds': preds[-1], '_probs': probs[-1]},
        {'_preds': preds, '_probs': probs},
    )
    assert results == aligned
    assert last == aligned[-1]
    assert sensor.predict(raw.head(2)).reason == 'warm-up'
    assert sensor.predict(raw.head(100)).reason == 'inside-training-window'
    assert sensor.predict(raw.head(389)).reason == 'null-features'
