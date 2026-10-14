from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol, TypeVar, cast

import polars as pl

from limen.backtest.trade_contract import finite_number
from limen.experiment._backtest_provenance import SOURCE_ROW
from limen.experiment._shim_fractional_diff import fractional_column, fit_order

__all__ = ['FittedColumnShims', 'apply_column_shims', 'parse_shims', 'prepare_column_shims', 'set_feature_shims']


class _ShimManifest(Protocol):
    feature_shims: list[dict[str, object]]

    @property
    def architecture_function(self) -> object: ...

    @property
    def target_column(self) -> str | None: ...

    def resolve_model_kwargs(self, round_params: dict[str, object]) -> Mapping[str, object]: ...


_TManifest = TypeVar('_TManifest', bound=_ShimManifest)

_METHOD_FIELDS: dict[str, set[str]] = {
    'ratio': {'reference'},
    'scale': {'reference'},
    'log1p': set(),
    'fractional_diff': {'d', 'threshold', 'significance_level'},
}
_FIELDS = {'columns', 'method', 'reference', 'd', 'threshold', 'significance_level'}


def _number(value: object, name: str) -> float:
    number = finite_number(value, f'feature_shims {name}')
    valid = (0 <= number <= 1 if name == 'd' else
             number > 0 if name == 'threshold' else 0 < number < 1)
    if not valid:
        raise ValueError(f'feature_shims {name} is outside its supported range')
    return number


def _is_reference(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r'\{\w+\}', value) is not None


def _spec(value: Mapping[str, object], *, references: bool) -> dict[str, object]:
    spec = dict(value)
    if set(spec) - _FIELDS or not {'columns', 'method'} <= set(spec):
        raise ValueError('feature_shims requires columns/method and only declared method fields')
    columns = spec['columns']
    if not isinstance(columns, list) or not columns:
        raise TypeError('feature_shims columns must be a nonempty list of names')
    if not all(isinstance(name, str) and name for name in cast(list[object], columns)):
        raise TypeError('feature_shims columns must contain nonempty strings')
    method = spec['method']
    if 'reference' in spec and (not isinstance(spec['reference'], str) or not spec['reference']):
        raise ValueError('feature_shims reference must be a nonempty column name')
    for name in ('d', 'threshold', 'significance_level'):
        if name in spec and not (references and _is_reference(spec[name])) and not (name == 'd' and spec[name] == 'auto'):
            spec[name] = _number(spec[name], name)
    if references and _is_reference(method):
        return spec
    if not isinstance(method, str) or method not in _METHOD_FIELDS:
        raise ValueError('feature_shims method must be ratio, scale, log1p or fractional_diff')
    if set(spec) - {'columns', 'method'} - _METHOD_FIELDS[method]:
        raise ValueError(f'feature_shims {method} has conflicting method fields')
    if method in ('ratio', 'scale'):
        reference = spec.get('reference')
        if not isinstance(reference, str) or not reference:
            raise ValueError(f'feature_shims {method} requires a reference column')
    if method == 'fractional_diff':
        _ = spec.setdefault('d', 'auto')
        _ = spec.setdefault('threshold', 0.001)
        _ = spec.setdefault('significance_level', 0.05)
        for name in ('d', 'threshold', 'significance_level'):
            field_value = spec[name]
            if (references and _is_reference(field_value)) or (name == 'd' and field_value == 'auto'):
                continue
            spec[name] = _number(field_value, name)
    return spec


def parse_shims(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise TypeError('feature_shims must be a list of declarations')
    result: list[dict[str, object]] = []
    selected: set[str] = set()
    for entry in cast(list[object], value):
        if not isinstance(entry, Mapping) or not all(
            isinstance(name, str) for name in cast(Mapping[object, object], entry)
        ):
            raise TypeError('feature_shims entries must be mappings with string fields')
        spec = _spec(cast(Mapping[str, object], entry), references=True)
        columns = spec['columns']
        if isinstance(columns, list):
            for column in cast(list[str], columns):
                if column in selected or column in {'datetime', SOURCE_ROW, '_source_row'}:
                    raise ValueError(f'feature_shims protected or overlapping column {column!r}')
                selected.add(column)
        result.append(spec)
    return copy.deepcopy(result)


def _format(value: str, params: Mapping[str, object], *, column: bool = False) -> object:
    try:
        return params[value[1:-1]] if not column and _is_reference(value) else value.format_map(params)
    except KeyError as exc:
        raise ValueError(f'feature_shims unknown parameter {exc.args[0]}') from exc


def _resolve(value: object, params: Mapping[str, object]) -> object:
    return _format(value, params) if isinstance(value, str) else value


def _resolved(specs: list[dict[str, object]], params: Mapping[str, object]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for spec in parse_shims(specs):
        values = {name: _resolve(value, params) for name, value in spec.items()}
        columns = values['columns']
        if isinstance(columns, list):
            values['columns'] = [
                _format(name, params, column=True) if isinstance(name, str) else name
                for name in cast(list[object], columns)
            ]
        result.append(_spec(values, references=False))
    return result


def _series(data: pl.DataFrame, column: str) -> pl.Series:
    if column not in data.columns or not data.schema[column].is_numeric():
        raise ValueError(f'feature_shims column {column!r} must exist and be numeric')
    values = data[column].cast(pl.Float64)
    if not values.drop_nulls().is_finite().all():
        raise ValueError(f'feature_shims column {column!r} contains nonfinite values')
    return values


def _validate(data: pl.DataFrame, specs: list[dict[str, object]], target: str | None) -> None:
    protected = {'datetime', SOURCE_ROW, '_source_row', target}
    selected: set[str] = set()
    for spec in specs:
        for column in cast(list[str], spec['columns']):
            if column in protected or column in selected:
                raise ValueError(f'feature_shims protected or overlapping column {column!r}')
            selected.add(column)
            _ = _series(data, column)
        if spec['method'] in ('ratio', 'scale'):
            reference = cast(str, spec['reference'])
            if reference in protected:
                raise ValueError(f'feature_shims protected reference {reference!r}')
            if (_series(data, reference).drop_nulls() == 0).any():
                raise ValueError(f'feature_shims reference {reference!r} contains zero')


def _fingerprint(specs: list[dict[str, object]], target: str | None) -> str:
    return sha256(json.dumps({'specs': specs, 'target': target}, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class FittedColumnShims:
    specs: list[dict[str, object]]
    orders: dict[str, float]
    target: str | None
    fingerprint: str

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        _validate(data, self.specs, self.target)
        replacements: list[pl.Series] = []
        for spec in self.specs:
            method = spec['method']
            for column in cast(list[str], spec['columns']):
                values = _series(data, column)
                if method == 'fractional_diff':
                    values = fractional_column(data, column, self.orders[column], cast(float, spec['threshold']))
                elif method == 'log1p':
                    if (values.drop_nulls() <= -1).any():
                        raise ValueError(f'feature_shims log1p column {column!r} requires values greater than -1')
                    values = values.log1p()
                else:
                    values = values / _series(data, cast(str, spec['reference']))
                    if method == 'ratio':
                        values = values - 1
                if not values.drop_nulls().is_finite().all():
                    raise ValueError(f'feature_shims output {column!r} is nonfinite')
                replacements.append(values.rename(column))
        return data.with_columns(replacements)


def apply_column_shims(
    data: pl.DataFrame, specs: list[dict[str, object]], round_params: Mapping[str, object],
    fitted: object | None, *, training: bool, target: str | None,
) -> tuple[pl.DataFrame, FittedColumnShims]:
    resolved = _resolved(specs, round_params)
    fingerprint = _fingerprint(resolved, target)
    _validate(data, resolved, target)
    if fitted is not None:
        if not isinstance(fitted, FittedColumnShims) or fitted.fingerprint != fingerprint:
            raise ValueError('feature_shims fitted state does not match the resolved declaration')
        state = fitted
    else:
        if not training:
            raise ValueError('feature_shims fitted state is required outside training')
        orders: dict[str, float] = {}
        for spec in resolved:
            if spec['method'] == 'fractional_diff':
                for column in cast(list[str], spec['columns']):
                    orders[column] = fit_order(
                        data, column, spec['d'], cast(float, spec['threshold']),
                        cast(float, spec['significance_level']),
                    )
        state = FittedColumnShims(copy.deepcopy(resolved), orders, target, fingerprint)
    return state.transform(data), state


def set_feature_shims(self: _TManifest, shims: list[dict[str, object]]) -> _TManifest:
    self.feature_shims = parse_shims(shims)
    return self


def prepare_column_shims(
    manifest: _ShimManifest, data: pl.DataFrame, round_params: Mapping[str, object],
    fitted_params: dict[str, object], training: bool,
) -> pl.DataFrame:
    if manifest.feature_shims:
        automatic = any(spec['method'] == 'fractional_diff' and spec.get('d') == 'auto'
                        for spec in _resolved(manifest.feature_shims, round_params))
        if automatic:
            from limen.sfd.reference_architecture.direction_sizing import direction_sizing

            if (manifest.architecture_function is direction_sizing
                    and manifest.resolve_model_kwargs(dict(round_params)).get('conditional_size') is True):
                raise ValueError('feature_shims auto fractional_diff is unsupported with conditional direction/sizing')
        data, fitted_params['_column_shims'] = apply_column_shims(
            data, manifest.feature_shims, round_params, fitted_params.get('_column_shims'),
            training=training, target=manifest.target_column,
        )
    return data.fill_nan(None)
