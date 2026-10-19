from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import polars as pl

from limen.experiment._column_shims import FittedColumnShims, apply_column_shims

__all__ = ['FoldCompression', 'FoldFeatures', 'FoldPreparation', 'FoldScaler', 'PreparedFolds', 'prepare_folds']


@dataclass(frozen=True)
class FoldFeatures:
    train: pl.DataFrame
    predict: pl.DataFrame
    train_rows: tuple[str, ...]
    predict_rows: tuple[str, ...]


@runtime_checkable
class FoldPreparation(Protocol):
    raw_features: pl.DataFrame
    row_ids: tuple[str, ...]
    identities: pl.DataFrame
    deterministic: bool
    def fit_transform(self, train_rows: Sequence[str], predict_rows: Sequence[str]) -> FoldFeatures: ...


@dataclass(frozen=True)
class PreparedFolds:
    raw_features: pl.DataFrame
    row_ids: tuple[str, ...]
    identities: pl.DataFrame
    deterministic: bool
    transform: Callable[[Sequence[str], Sequence[str]], FoldFeatures]

    def fit_transform(self, train_rows: Sequence[str], predict_rows: Sequence[str]) -> FoldFeatures:
        for requested in (train_rows, predict_rows):
            requested_ids = set(requested)
            if len(requested_ids) != len(requested) or tuple(row for row in self.row_ids if row in requested_ids) != tuple(requested):
                raise ValueError('Fold preprocessing requires ordered original row identities')
        features = self.transform(train_rows, predict_rows)
        for frame, rows, requested in ((features.train, features.train_rows, train_rows),
                                       (features.predict, features.predict_rows, predict_rows)):
            returned_ids = set(rows)
            if frame.height != len(rows) or len(returned_ids) != len(rows) or tuple(row for row in requested if row in returned_ids) != rows:
                raise ValueError('Fold preprocessing changed causal feature-valid membership')
        return features


class FoldScaler(Protocol):
    def __call__(self, data: pl.DataFrame, *, all_fitted_params: dict[str, object], is_training: bool) -> tuple[pl.DataFrame, dict[str, object]]: ...


class FoldCompression(Protocol):
    def __call__(self, split_data: list[pl.DataFrame], *, all_fitted_params: dict[str, object]) -> tuple[list[pl.DataFrame], dict[str, object]]: ...


def _valid(frame: pl.DataFrame) -> pl.DataFrame:
    numeric = [name for name, dtype in frame.schema.items() if dtype.is_numeric()]
    if numeric and frame.select(pl.any_horizontal(pl.col(numeric).is_infinite()).any()).item():
        raise ValueError('Fold preprocessing produced nonfinite features')
    return frame.fill_nan(None).drop_nulls()


def _scaled(train: pl.DataFrame, predict: pl.DataFrame, scale: FoldScaler, compress: FoldCompression, *, available: bool) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not train.height:
        raise ValueError('Fold preprocessing has no feature-valid training rows')
    fitted: dict[str, object] = {}
    source = train
    train, fitted = scale(train, all_fitted_params=fitted, is_training=True)
    context_rows = max((int(getattr(value, 'context_rows', 0)) for value in fitted.values()), default=0)
    prefix = source.tail(context_rows)
    predict, fitted = scale(pl.concat([prefix, predict]), all_fitted_params=fitted, is_training=False)
    predict = predict.slice(prefix.height)
    if available:
        train, predict = _valid(train), _valid(predict)
    transformed, _ = compress([train, predict, predict], all_fitted_params=fitted)
    names = [name for name in transformed[0].columns if name != 'datetime']
    if any(transformed[index].select(names).null_count().sum_horizontal()[0] for index in (0, 1)):
        raise ValueError('Fold preprocessing has unavailable features; provide sufficient causal context')
    return transformed[0], transformed[1]


def _result(train: pl.DataFrame, predict: pl.DataFrame, identities: pl.DataFrame) -> FoldFeatures:
    rows = [tuple(str(value) for value in frame.select('datetime').join(identities, on='datetime', how='left', maintain_order='left')['row_id']) for frame in (train, predict)]
    return FoldFeatures(train.drop('datetime'), predict.drop('datetime'), rows[0], rows[1])


def prepare_folds(raw: pl.DataFrame, retained: pl.DataFrame, identities: pl.DataFrame, *, scale: FoldScaler, compress: FoldCompression, deterministic: bool, shims: FittedColumnShims | None = None, dropped: Sequence[str] = ()) -> PreparedFolds:
    automatic = shims is not None and any(spec['method'] == 'fractional_diff' and spec.get('d') == 'auto' for spec in shims.specs)
    if automatic:
        selected = _valid(raw.drop(dropped))
    else:
        transformed = raw if shims is None else shims.transform(raw)
        selected = retained.select('datetime').join(transformed.drop(dropped), on='datetime', how='left', maintain_order='left')
    mapping = selected.select('datetime').join(identities.select('datetime', 'row_id', 'available_at_ns'), on='datetime', how='left', maintain_order='left')
    if mapping['row_id'].null_count() or mapping['row_id'].is_duplicated().any():
        raise ValueError('Fold preprocessing requires unambiguous original row identities')
    row_ids = tuple(str(value) for value in mapping['row_id'])
    positions = {identity: index for index, identity in enumerate(row_ids)}
    original = raw.select('datetime').join(identities.select('datetime', 'row_id', 'available_at_ns'), on='datetime', how='left', maintain_order='left')
    original_positions = {str(identity): index for index, identity in enumerate(original['row_id'])}

    def transform(train_rows: Sequence[str], predict_rows: Sequence[str]) -> FoldFeatures:
        if automatic:
            if shims is None:
                raise ValueError('Automatic fold preprocessing requires its shim declaration')
            if not train_rows or not predict_rows or original_positions[train_rows[-1]] >= original_positions[predict_rows[0]]:
                raise ValueError('Fold preprocessing requires a nonempty earlier training prefix')
            cutoff = mapping['available_at_ns'][positions[predict_rows[0]]]
            history = raw.filter(original['available_at_ns'] < cutoff)
            _, state = apply_column_shims(history, shims.specs, {}, None, training=True, target=shims.target)
            transformed = state.transform(raw.head(original_positions[predict_rows[-1]] + 1)).drop(dropped)
            train_times = mapping.filter(pl.col('row_id').is_in(train_rows)).select('datetime')
            predict_times = mapping.filter(pl.col('row_id').is_in(predict_rows)).select('datetime')
            train = _valid(train_times.join(transformed, on='datetime', how='left', maintain_order='left'))
            predict = _valid(predict_times.join(transformed, on='datetime', how='left', maintain_order='left'))
        else:
            train = selected[[positions[row] for row in train_rows]]
            predict = selected[[positions[row] for row in predict_rows]]
        train, predict = _scaled(train, predict, scale, compress, available=automatic)
        return _result(train, predict, mapping)

    return PreparedFolds(selected.drop('datetime'), row_ids, mapping, deterministic, transform)
