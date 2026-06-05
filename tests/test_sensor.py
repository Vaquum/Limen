from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import polars as pl

from limen.experiment import MLManifest
from limen.experiment.manifest_core import PCACompressionConfig
from limen.experiment.manifest_core import _apply_sensor_pca
from limen.experiment.manifest_core import _count_leading_nulls
from limen.experiment.trainer.sensor import BarPrediction
from limen.experiment.trainer.sensor import Sensor
from limen.experiment.trainer.sensor import _extract_scalar
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


def test_extract_scalar_rejects_bool() -> None:
    assert _extract_scalar(True) is None


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
