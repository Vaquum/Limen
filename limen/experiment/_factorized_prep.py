"""Invocation-local reuse of fitted preparation artifacts, never prepared data."""

from __future__ import annotations

import copy
import inspect
import json
import warnings
from collections.abc import Callable, Mapping
from typing import cast

import polars as pl

from limen.experiment.manifest_core import MLManifest
from limen.experiment._factorized_admission import supports_preparation
from limen.scalers.robust_scaler import RobustScaler
from limen.targets.next_return import NextReturnTarget
from limen.targets.quantile_binary import QuantileBinaryTarget

__all__ = ['FittedPreparation', 'supports_preparation']


class FittedPreparation:
    def __init__(self, manifest: MLManifest, domain: Mapping[str, list[object]], axes: frozenset[str],
                 references: Callable[[object, str], bool]) -> None:
        super().__init__()
        self.manifest = manifest
        if manifest.architecture_function is None:
            raise ValueError('Fitted preparation requires an architecture')
        consumers = [value for attr, value in vars(manifest).items()
                     if attr not in ('backtest_config', 'architecture_function', 'architecture_params')]
        self.names = frozenset(name for name in domain if name not in axes and (
            name not in inspect.signature(manifest.architecture_function).parameters or any(references(value, name) for value in consumers)))
        self.cache: dict[str, dict[str, object]] = {}
        self.warning_cache: dict[str, list[warnings.WarningMessage]] = {}
        self.pending_warnings: list[warnings.WarningMessage] = []

    def prepare(self, raw: pl.DataFrame, params: dict[str, object],
                prep: Callable[..., dict[str, object]]) -> tuple[dict[str, object], str, dict[str, object]]:
        key = json.dumps({name: params[name] for name in self.names}, sort_keys=True, default=str)
        saved = self.cache.pop(key, None)
        if saved is not None:
            self.cache[key] = saved
        prepare_fitted = cast(Callable[..., dict[str, object]], vars(MLManifest)['_prepare_data_with_fitted'])
        caught: list[warnings.WarningMessage] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                data = (prep(raw, round_params=params) if saved is None else
                        prepare_fitted(self.manifest, raw, params, copy.deepcopy(saved)))
        finally:
            for warning in [*(self.warning_cache[key] if saved is not None else []), *caught]:
                warnings.warn(str(warning.message), warning.category, stacklevel=2)
        self.pending_warnings = list(caught) if saved is None else self.warning_cache[key]
        artifacts = data.get('_fitted_params')
        if not isinstance(artifacts, dict):
            raise ValueError('Factorized preparation requires fitted artifact evidence')
        fitted = cast(dict[str, object], artifacts)
        if any(type(value) not in (RobustScaler, NextReturnTarget, QuantileBinaryTarget)
               for value in fitted.values()):
            raise ValueError('Factorized preparation contains an unsupported fitted artifact')
        return data, key, copy.deepcopy(fitted)

    def evidence(self, artifacts: dict[str, object]) -> dict[str, object]:
        return copy.deepcopy({'_fitted_params': artifacts, **artifacts})
