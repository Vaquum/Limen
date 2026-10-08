from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import ClassVar, Literal, Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.linear_model import LogisticRegression, Ridge
from typing_extensions import override

from limen.backtest.trade_contract import JsonValue, TradePolicy, finite_number, json_value, contract_digest, export_trade_contract, source_binding
from limen.experiment._prepare_trade_context import PreparedTradeContext
from limen.sfd.reference_architecture.base import ReferenceModel
from limen.targets.trade_outcome import OutcomeLabels

Array = npt.NDArray[np.float64]


class ComponentEstimator(Protocol):
    def fit(self, x: Array, y: Array) -> object: ...
    def predict(self, x: Array) -> Array: ...


class ComponentFactory(Protocol):
    deterministic: bool
    def __call__(self, *, seed: int, **params: JsonValue) -> ComponentEstimator: ...


def _direction_factory(*, seed: int, **params: JsonValue) -> ComponentEstimator:
    return cast(Callable[..., ComponentEstimator], LogisticRegression)(random_state=seed, **params)


def _sizing_factory(*, seed: int, **params: JsonValue) -> ComponentEstimator:
    return cast(Callable[..., ComponentEstimator], Ridge)(random_state=seed, **params)


setattr(_direction_factory, 'deterministic', True)
setattr(_sizing_factory, 'deterministic', True)


@dataclass(frozen=True)
class FoldFeatures:
    train: pl.DataFrame
    predict: pl.DataFrame


@runtime_checkable
class FoldPreparation(Protocol):
    raw_features: pl.DataFrame
    row_ids: tuple[str, ...]
    deterministic: bool
    def fit_transform(self, train_rows: Sequence[str], predict_rows: Sequence[str]) -> FoldFeatures: ...


def _identity(factory: object) -> dict[str, JsonValue]:
    module_name, name = getattr(factory, '__module__', None), getattr(factory, '__qualname__', None)
    if not isinstance(module_name, str) or not isinstance(name, str) or '<locals>' in name:
        raise ValueError('Component factories require importable module-qualified identities')
    module = importlib.import_module(module_name)
    resolved: object = module
    for part in name.split('.'):
        resolved = getattr(resolved, part)
    if resolved is not factory or module.__file__ is None:
        raise ValueError('Component factory identity does not resolve to its original source')
    return {'reference': f'{module_name}.{name}', 'source_sha256': sha256(Path(module.__file__).read_bytes()).hexdigest()}


def _labels(data: Mapping[str, object], index: int, scale: float, maximum: float) -> tuple[pl.DataFrame, Array, Array, npt.NDArray[np.bool_]]:
    labels, context = data.get('_trade_labels'), data.get('_trade_context')
    if not isinstance(labels, OutcomeLabels) or not isinstance(context, PreparedTradeContext):
        raise ValueError('Direction/sizing requires private bound simulated-trade labels')
    rows = context.partitions[index].signals.select('row_id', 'available_at_ns').join(labels.rows, on='row_id', how='left', maintain_order='left')
    if rows['long_available'].null_count() or rows['short_available'].null_count():
        raise ValueError('Private label identities do not cover the causal inference rows')
    long = rows['long_return'].to_numpy().astype(np.float64)
    short = rows['short_return'].to_numpy().astype(np.float64)
    valid = rows['long_available'].to_numpy() & rows['short_available'].to_numpy()
    direction = np.where((long > 0) & (long > short), 1.0, np.where((short > 0) & (short > long), -1.0, 0.0))
    size = np.where(direction == 0, 0.0, np.clip(np.maximum(long, short) / scale, 0, maximum))
    return rows, direction, size, valid


def _columns(names: Sequence[str] | None, features: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(features if names is None else names)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(features):
        raise ValueError('Component feature subsets must be unique prepared feature names')
    return selected


def _predict(estimator: ComponentEstimator, x: Array, *, direction: bool = False) -> Array:
    result = np.asarray(estimator.predict(x), dtype=np.float64)
    if result.shape != (x.shape[0],) or not np.isfinite(result).all():
        raise ValueError('Component predictions must be finite scalars for every input row')
    if direction and not np.isin(result, [-1, 0, 1]).all():
        raise ValueError('Direction component must predict -1, 0 or +1')
    return result


class DirectionSizingModel(ReferenceModel):
    prediction_mode: ClassVar[Literal['binary', 'target_exposure']] = 'target_exposure'
    direction_model: ComponentEstimator
    sizing_model: ComponentEstimator

    @override
    def train(self, data: dict[str, object], **params: object) -> 'DirectionSizingModel':
        config = dict(params)
        allowed = {'direction_factory', 'sizing_factory', 'direction_params', 'sizing_params', 'direction_features', 'sizing_features', 'conditional_size', 'folds', 'min_train_samples', 'return_scale', 'max_size', 'seed'}
        if set(config) - allowed:
            raise ValueError(f'Unknown direction/sizing params: {sorted(set(config) - allowed)}')
        self.scale = finite_number(config.get('return_scale', 0.01), 'return scale')
        self.maximum = finite_number(config.get('max_size', 1.0), 'maximum size')
        if self.scale <= 0 or not 0 < self.maximum <= 1:
            raise ValueError('Return scale must be positive and maximum size in (0, 1]')
        seed, folds, minimum = config.get('seed', 42), config.get('folds', 5), config.get('min_train_samples', 20)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (seed, folds, minimum)) or cast(int, folds) < 2 or cast(int, minimum) < 2:
            raise ValueError('Seed/folds/minimum require integers, folds/minimum at least two')
        conditional = config.get('conditional_size', False)
        if not isinstance(conditional, bool):
            raise ValueError('conditional_size must be boolean')
        self.conditional = conditional
        direction_factory = cast(ComponentFactory, config.get('direction_factory', _direction_factory))
        sizing_factory = cast(ComponentFactory, config.get('sizing_factory', _sizing_factory))
        direction_params = cast(Mapping[str, JsonValue], config.get('direction_params') or {})
        sizing_params = cast(Mapping[str, JsonValue], config.get('sizing_params') or {})
        train = data['x_train']
        if not isinstance(train, pl.DataFrame):
            raise ValueError('Direction/sizing training requires named prepared features')
        self.feature_names = tuple(train.columns)
        self.direction_features = _columns(cast(Sequence[str] | None, config.get('direction_features')), self.feature_names)
        self.sizing_features = _columns(cast(Sequence[str] | None, config.get('sizing_features')), self.feature_names)
        policy = data.get('_trade_policy')
        if not isinstance(policy, TradePolicy) or policy.prediction_mode != 'target_exposure':
            raise ValueError('Direction/sizing requires target-exposure execution economics')
        self.flat_threshold = policy.flat_threshold
        rows, direction, size, valid = _labels(data, 0, self.scale, self.maximum)
        if valid.sum() < cast(int, minimum) or np.unique(direction[valid]).size < 2:
            raise ValueError('Insufficient completed direction/sizing samples or direction classes')
        self.learning_binding = {'architecture': _identity(direction_sizing), 'direction_factory': _identity(direction_factory), 'sizing_factory': _identity(sizing_factory), 'direction_params': json_value(direction_params), 'sizing_params': json_value(sizing_params), 'direction_features': list(self.direction_features), 'sizing_features': list(self.sizing_features), 'conditional_size': self.conditional, 'folds': folds, 'min_train_samples': minimum, 'return_scale': self.scale, 'max_size': self.maximum, 'seed': seed, 'label_mapping': 'best_positive_side_ties_flat_v1', 'trade_contract_digest': json_value(data.get('trade_contract_digest'))}
        context = data.get('_trade_context')
        if not isinstance(context, PreparedTradeContext):
            raise ValueError('Missing frozen training source contracts')
        self.learning_binding['source_contracts'] = json_value([contract_digest(export_trade_contract(context.policy, partition)) for partition in context.partitions])
        self.learning_binding['feature_sources'] = json_value([source_binding(cast(pl.DataFrame, data[key]), key, 0, 0, 1, 'prepared_feature_binding').checksum for key in ('x_train', 'x_val', 'x_test')])
        self.direction_model = direction_factory(seed=cast(int, seed), **direction_params)
        self.sizing_model = sizing_factory(seed=cast(int, seed), **sizing_params)
        fit_size = valid.copy()
        size_x = train.select(self.sizing_features).to_numpy().astype(np.float64)
        if self.conditional:
            oof = self._oof(data, rows, direction, valid, direction_factory, direction_params, cast(int, seed), cast(int, folds), cast(int, minimum))
            fit_size &= np.isfinite(oof)
            size_x = np.column_stack((size_x, oof))
        if fit_size.sum() < cast(int, minimum):
            raise ValueError('Insufficient available sizing samples after chronological overlap purging')
        _ = self.direction_model.fit(train.select(self.direction_features).to_numpy().astype(np.float64)[valid], direction[valid])
        _ = self.sizing_model.fit(size_x[fit_size], size[fit_size])
        fold_preparation = data.get('_fold_preparation')
        self.deterministic = bool(direction_factory.deterministic and sizing_factory.deterministic and isinstance(fold_preparation, FoldPreparation) and fold_preparation.deterministic)
        data['_learning_binding'] = self.learning_binding
        alignment = data.get('_alignment')
        if not isinstance(alignment, dict):
            raise ValueError('Learning identity requires round alignment metadata')
        cast(dict[str, object], alignment)['learning_binding'] = self.learning_binding
        return self

    def _oof(self, data: Mapping[str, object], rows: pl.DataFrame, direction: Array, valid: npt.NDArray[np.bool_], factory: ComponentFactory, params: Mapping[str, JsonValue], seed: int, folds: int, minimum: int) -> Array:
        preparation = data.get('_fold_preparation')
        if not isinstance(preparation, FoldPreparation) or tuple(rows['row_id']) != preparation.row_ids:
            raise ValueError('Conditional sizing requires its original chronological fold preparation')
        if folds > rows.height:
            raise ValueError('Fold count exceeds the causal training population')
        oof = np.full(rows.height, np.nan, dtype=np.float64)
        availability = rows['available_at_ns'].to_numpy()
        completion = rows.select(pl.max_horizontal('long_label_available_ns', 'short_label_available_ns').fill_null(0)).to_series().to_numpy()
        for fold, block in enumerate(np.array_split(np.arange(rows.height), folds)):
            if fold == 0:
                continue
            start = availability[block[0]]
            earlier = np.flatnonzero(availability < start)
            fit = earlier[valid[earlier] & (completion[earlier] < start)]
            if fit.size < minimum or np.unique(direction[fit]).size < 2:
                raise ValueError(f'Chronological fold {fold}: insufficient purged samples/classes')
            train_ids = [preparation.row_ids[index] for index in earlier]
            predict_ids = [preparation.row_ids[index] for index in block]
            features = preparation.fit_transform(train_ids, predict_ids)
            names = _columns(self.direction_features, features.train.columns)
            model = factory(seed=seed, **params)
            positions = {int(index): local for local, index in enumerate(earlier)}
            local = [positions[int(index)] for index in fit]
            _ = model.fit(features.train.select(names).to_numpy().astype(np.float64)[local], direction[fit])
            oof[block] = _predict(model, features.predict.select(names).to_numpy().astype(np.float64), direction=True)
        return oof

    def _components(self, value: object) -> tuple[Array, Array]:
        if isinstance(value, pl.DataFrame):
            x = value.select(self.feature_names).to_numpy().astype(np.float64)
        else:
            x = np.asarray(value, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names) or not np.isfinite(x).all():
            raise ValueError('Inference features do not match the fitted component bindings')
        direction = _predict(self.direction_model, x[:, [self.feature_names.index(name) for name in self.direction_features]], direction=True)
        sizing = x[:, [self.feature_names.index(name) for name in self.sizing_features]]
        if self.conditional:
            sizing = np.column_stack((sizing, direction))
        size = np.clip(_predict(self.sizing_model, sizing), 0, self.maximum)
        return direction, size

    @override
    def predict(self, data: dict[str, object]) -> dict[str, object]:
        direction, size = self._components(data['x_test'])
        exposure = direction * size
        exposure[np.abs(exposure) <= self.flat_threshold] = 0
        return {'_preds': exposure}

    @override
    def evaluate(self, data: dict[str, object], inline_metrics: bool = True) -> dict[str, object]:
        direction, size = self._components(data['x_test'])
        result = self.predict(data)
        _, expected_direction, expected_size, valid = _labels(data, 2, self.scale, self.maximum)
        result['direction_accuracy'] = float(np.mean(direction[valid] == expected_direction[valid])) if valid.any() else None
        result['size_mae'] = float(np.mean(np.abs(size[valid] - expected_size[valid]))) if valid.any() else None
        if inline_metrics:
            result.update(self._compute_backtest(cast(Array, result['_preds']), data))
        return result


def direction_sizing(data: dict[str, object], *, direction_factory: ComponentFactory = cast(ComponentFactory, _direction_factory), sizing_factory: ComponentFactory = cast(ComponentFactory, _sizing_factory), direction_params: Mapping[str, JsonValue] | None = None, sizing_params: Mapping[str, JsonValue] | None = None, direction_features: Sequence[str] | None = None, sizing_features: Sequence[str] | None = None, conditional_size: bool = False, folds: int = 5, min_train_samples: int = 20, return_scale: float = 0.01, max_size: float = 1.0, seed: int = 42, inline_metrics: bool = True) -> dict[str, object]:
    params = dict(locals())
    del params['data'], params['inline_metrics']
    model = DirectionSizingModel().train(data, **params)
    result = model.evaluate(data, inline_metrics=inline_metrics)
    result['_model'] = model
    return result


setattr(direction_sizing, 'requires_trade_outcomes', True)
__all__ = ['ComponentEstimator', 'ComponentFactory', 'DirectionSizingModel', 'FoldFeatures', 'FoldPreparation', 'direction_sizing']
