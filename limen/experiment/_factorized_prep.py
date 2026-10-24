"""Invocation-local reuse of fitted preparation artifacts, never prepared data."""

from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Callable, Mapping
from types import FunctionType
from typing import cast

import polars as pl

import limen.experiment._holdout_composition as _composition
from limen.experiment.manifest_core import MLManifest, make_fitted_scaler
from limen.features.lagged_features import lag_range
from limen.indicators.window_return import window_return
from limen.scalers.robust_scaler import RobustScaler
from limen.targets.next_return import NextReturnTarget
from limen.targets.quantile_binary import QuantileBinaryTarget

__all__ = ['FittedPreparation', 'supports_preparation']


def _supports_scaler(manifest: MLManifest) -> bool:
    if manifest.scaler is None:
        return True
    computations, transform, params = manifest.scaler
    if len(computations) != 1:
        return False
    name, factory, dynamic = computations[0]
    reference = make_fitted_scaler(name, RobustScaler)
    reference_factory = reference[0][0][1]
    return (isinstance(factory, FunctionType) and isinstance(reference_factory, FunctionType)
            and factory.__code__ is reference_factory.__code__ and transform is reference[1]
            and inspect.signature(factory).parameters['_cls'].default is RobustScaler
            and set(dynamic).issubset({'quantile_range'})
            and all(isinstance(value, str) for value in dynamic.values())
            and params == {'fitted_transform': name})


def supports_preparation(manifest: MLManifest) -> bool:
    target = manifest.target_class_config
    return (target is not None and target.target_class in (NextReturnTarget, QuantileBinaryTarget)
            and _supports_scaler(manifest) and manifest.ablation_config is None
            and manifest.pca_compression_config is None and manifest.prediction_calibration_config is None
            and not manifest.feature_shims
            and manifest.data_dict_extension is None and manifest.objective is None
            and all(entry.func in (window_return, lag_range) for entry in manifest.feature_transforms))


class FittedPreparation:
    def __init__(self, manifest: MLManifest, domain: Mapping[str, list[object]], axes: frozenset[str],
                 references: Callable[[object, str], bool]) -> None:
        self.manifest = manifest
        if manifest.architecture_function is None:
            raise ValueError('Fitted preparation requires an architecture')
        consumers = [value for attr, value in vars(manifest).items()
                     if attr not in ('backtest_config', 'architecture_function', 'architecture_params')]
        self.names = frozenset(name for name in domain if name not in axes and (
            name not in inspect.signature(manifest.architecture_function).parameters or any(references(value, name) for value in consumers)))
        self.cache: dict[str, dict[str, object]] = {}
        self.proofs: dict[str, dict[str, object]] = {}

    def prepare(self, raw: pl.DataFrame, params: dict[str, object],
                prep: Callable[..., dict[str, object]]) -> tuple[dict[str, object], str, dict[str, object]]:
        key = json.dumps({name: params[name] for name in self.names}, sort_keys=True, default=str)
        saved = _composition.prepared_cache(self.cache, self.proofs, self.manifest, raw, key)
        prepare_fitted = cast(Callable[..., dict[str, object]], vars(MLManifest)['_prepare_data_with_fitted'])
        data = (prep(raw, round_params=params) if saved is None else
                prepare_fitted(self.manifest, raw, params, copy.deepcopy(saved)))
        artifacts = data.get('_fitted_params')
        if not isinstance(artifacts, dict):
            raise ValueError('Factorized preparation requires fitted artifact evidence')
        fitted = _composition.fitted_artifacts(self.manifest, data, cast(dict[str, object], artifacts),
                                                self.proofs.get(key) if saved is not None else None)
        if any(type(value) not in (RobustScaler, NextReturnTarget, QuantileBinaryTarget)
               for value in fitted.values()):
            raise ValueError('Factorized preparation contains an unsupported fitted artifact')
        return data, key, copy.deepcopy(fitted)

    def evidence(self, artifacts: dict[str, object]) -> dict[str, object]:
        return copy.deepcopy({'_fitted_params': artifacts, **artifacts})
