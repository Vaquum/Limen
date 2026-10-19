from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING
import polars as pl

if TYPE_CHECKING:
    from limen.experiment.manifest_core import MLManifest


def sensor_input_prep(manifest: MLManifest, raw_klines: pl.DataFrame,
                      fitted_params: Mapping[str, object], round_params: dict[str, object]) -> tuple[pl.DataFrame, int]:
    from limen.experiment.manifest_core import (_process_bars, _source_interval, _resolve_trade_number, sensor_decisions,
        _apply_feature_transforms, _drop_target_source_columns, _count_leading_nulls, _apply_scaler, _apply_sensor_pca)
    _, data = _process_bars(manifest, raw_klines, round_params)
    decisions = None
    if manifest.resolve_trade_policy(round_params) is not None:
        interval = _source_interval(manifest.data_source_config)
        interval = None if interval is None else _resolve_trade_number(interval, round_params, 'recorded source interval')
        decisions = sensor_decisions(raw_klines, data, interval_seconds=interval)
        data = data.drop([name for name in ('start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns') if name in data.columns])

    lazy = data.lazy()
    lazy = _apply_feature_transforms(manifest, lazy, round_params)
    data = lazy.collect()
    if manifest.target_class_config is not None:
        target = fitted_params.get(f'_target_cls_{manifest.target_column}', manifest.target_class_config.target_class)
        data = _drop_target_source_columns(data, target, manifest.target_column)

    data = manifest._prepare_column_shims(data, round_params, dict(fitted_params), False)
    if dropped_features := round_params.get('_dropped_features'):
        data = data.drop([c for c in dropped_features if c in data.columns])

    indicator_lookback = _count_leading_nulls(data)

    all_fitted_params = dict(fitted_params)
    data, _ = _apply_scaler(manifest, data, round_params, all_fitted_params, is_training=False)
    data = data.fill_nan(None)

    data = _apply_sensor_pca(manifest, data, round_params, all_fitted_params)

    if decisions is not None:
        data = data.join(decisions, on='datetime', how='left', maintain_order='left')
    return data, indicator_lookback

