from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

from limen.experiment._resolve_backtest_config import BACKTEST_KEYS, validate_backtest_value as _validate_backtest_value
from limen.yaml.errors import YAMLError
from limen.experiment._resolve_trade_policy import FundingConfig, ProductConfig, TRADE_NUMBERS
from limen.backtest.trade_contract import BPS, finite_number

if TYPE_CHECKING:
    from limen.experiment.manifest_core import Manifest

_PARAM_REF_RE = re.compile(r'\{(\w+)\}')


def _mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def check_backtest_spec(yaml_dict: Mapping[str, object], errors: list[YAMLError]) -> None:
    sfd = _mapping(yaml_dict.get('sfd'))
    backtest = _mapping(_mapping(sfd.get('manifest')).get('backtest'))
    params = _mapping(sfd.get('params'))
    allocation = _mapping(_mapping(sfd.get('manifest')).get('allocation'))
    if allocation:
        _check_source(allocation, 'sfd.manifest.allocation', errors)
        if backtest.get('prediction_mode') != 'target_exposure' or backtest.get('product') is None:
            errors.append(YAMLError(message='Allocation requires explicit event target-exposure economics', path='sfd.manifest.allocation'))
    for key in BACKTEST_KEYS:
        if key in backtest:
            value = backtest[key]
            path = f'sfd.manifest.backtest.{key}'
            if isinstance(value, str):
                ref = _PARAM_REF_RE.fullmatch(value.strip())
                if ref is None or ref.group(1) not in params:
                    errors.append(YAMLError(message=f"backtest.{key} requires a number or a {{param}} reference in sfd.params (got '{value}')", path=path))
                else:
                    candidates = params[ref.group(1)]
                    if isinstance(candidates, list):
                        for index, candidate in enumerate(cast(list[object], candidates)):
                            _check_value(key, candidate, f'{path} / sfd.params.{ref.group(1)}[{index}]', errors)
            else:
                _check_value(key, value, path, errors)
    for name, config in (('product', ProductConfig), ('funding', FundingConfig)):
        if backtest.get(name) is not None:
            value = backtest[name]
            if not isinstance(value, Mapping):
                errors.append(YAMLError(message=f'{name} must be a mapping', path=f'sfd.manifest.backtest.{name}'))
            else:
                unknown = set(_mapping(cast(object, value))) - set(cast(Mapping[str, object], config.__dataclass_fields__))
                if unknown:
                    errors.append(YAMLError(message=f'Unknown {name} keys: {sorted(unknown)}', path=f'sfd.manifest.backtest.{name}'))
    for key in ('execution_data_source', 'barrier_features'):
        _check_source(backtest.get(key), f'sfd.manifest.backtest.{key}', errors)
    funding = _mapping(backtest.get('funding'))
    _check_source(funding.get('data_source'), 'sfd.manifest.backtest.funding.data_source', errors)
    numeric: dict[str, object] = {name: backtest[name] for name in (*TRADE_NUMBERS, 'initial_equity') if name in backtest}
    product = _mapping(backtest.get('product'))
    if product:
        required = {'kind', 'instrument', 'base_currency', 'quote_currency', 'quantity_step', 'min_notional'}
        if required - set(product) or product.get('kind') not in ('cash_spot', 'linear_perpetual') or any(not isinstance(product.get(key), str) or not product.get(key) for key in ('instrument', 'base_currency', 'quote_currency')):
            errors.append(YAMLError(message='Product requires supported accounting, instrument, currencies and quantity rules', path='sfd.manifest.backtest.product'))
    numeric.update({name: product[name] for name in ('quantity_step', 'min_notional', 'initial_margin_fraction', 'maintenance_margin_fraction') if name in product})
    for name, value in numeric.items():
        if isinstance(value, str):
            ref = _PARAM_REF_RE.fullmatch(value)
            if ref is None or ref.group(1) not in params:
                errors.append(YAMLError(message=f'{name} requires a number or known {{param}} reference', path='sfd.manifest.backtest'))
            else:
                candidates = params[ref.group(1)]
                if isinstance(candidates, list):
                    for candidate in cast(list[object], candidates):
                        _check_trade_number(name, candidate, errors)
        else:
            _check_trade_number(name, value, errors)
    columns = [backtest.get(f'{side}_bps_column') for side in ('take_profit', 'stop_loss')]
    dynamic = any(value is not None for value in columns) or backtest.get('barrier_features') is not None
    if dynamic:
        if not any(columns) or backtest.get('barrier_features') is None or not product:
            errors.append(YAMLError(message='Dynamic barriers require distance columns, barrier_features and explicit product', path='sfd.manifest.backtest'))
        for side, column in zip(('take_profit', 'stop_loss'), columns, strict=True):
            if column is not None and (not isinstance(column, str) or not column.isidentifier() or column.startswith('__') or backtest.get(f'{side}_bps') is not None):
                errors.append(YAMLError(message=f'{side} requires a literal column or scalar, never both', path='sfd.manifest.backtest'))
        if columns[0] is not None and columns[0] == columns[1]:
            errors.append(YAMLError(message='Barrier distance columns must be distinct', path='sfd.manifest.backtest'))
    if backtest.get('source_interval_seconds') is not None:
        value = backtest['source_interval_seconds']
        if isinstance(value, str):
            errors.append(YAMLError(message='source_interval_seconds must be literal metadata', path='sfd.manifest.backtest.source_interval_seconds'))
        else:
            _check_trade_number('source_interval_seconds', value, errors)
    if backtest.get('prediction_mode', 'binary') not in ('binary', 'target_exposure'):
        errors.append(YAMLError(message='Unknown prediction mode', path='sfd.manifest.backtest.prediction_mode'))


def _check_source(value: object, path: str, errors: list[YAMLError]) -> None:
    if value is not None and (not isinstance(value, Mapping) or set(_mapping(cast(object, value))) - {'method', 'params'} or not isinstance(_mapping(cast(object, value)).get('method'), str) or not isinstance(_mapping(cast(object, value)).get('params', {}), Mapping)):
        errors.append(YAMLError(message='Source requires method and optional params only', path=path))


def _check_trade_number(name: str, value: object, errors: list[YAMLError]) -> None:
    if value is None and name in ('max_holding_seconds', 'timer_interval_seconds', 'max_price_gap_seconds'):
        return
    try:
        number = finite_number(value, name)
        positive = {'initial_equity', 'quantity_step', 'max_holding_seconds', 'timer_interval_seconds', 'max_price_gap_seconds', 'source_interval_seconds'}
        fractions = {'max_exposure', 'initial_margin_fraction'}
        if (name in positive and number <= 0) or (name in fractions and not 0 < number <= 1) or (name not in positive | fractions | {'timer_phase_utc_seconds'} and number < 0) or (name in ('flat_threshold', 'maintenance_margin_fraction') and number >= 1) or (name == 'signal_change_bps' and number >= BPS):
            raise ValueError(f'Invalid {name} range')
    except ValueError as exc:
        errors.append(YAMLError(message=str(exc), path=f'sfd.manifest.backtest.{name}'))


def _check_value(key: str, value: object, path: str, errors: list[YAMLError]) -> None:
    try:
        _ = _validate_backtest_value(key, value)
    except ValueError as exc:
        errors.append(YAMLError(message=str(exc), path=path))


__all__ = ['check_backtest_spec']


def apply_backtest(manifest: Manifest, declaration: Mapping[str, object]) -> None:
    from limen.experiment.manifest_core import DataSourceConfig
    from limen.experiment._resolve_trade_policy import BACKTEST_FIELDS
    from limen.yaml.resolver import resolve

    backtest = _mapping(declaration.get('backtest'))
    if not backtest:
        return
    kwargs = {key: backtest[key] for key in BACKTEST_FIELDS if key in backtest}
    if kwargs.get('product') is not None:
        kwargs['product'] = cast(Callable[..., object], ProductConfig)(**_mapping(kwargs['product']))
    for key in ('execution_data_source', 'barrier_features'):
        if kwargs.get(key) is not None:
            source = _mapping(kwargs[key])
            kwargs[key] = DataSourceConfig(resolve(cast(str, source['method'])), dict(_mapping(source.get('params'))))
    if kwargs.get('funding') is not None:
        funding = dict(_mapping(kwargs['funding']))
        if 'adapter' in funding:
            funding['adapter'] = resolve(cast(str, funding['adapter']))
        if funding.get('data_source') is not None:
            source = _mapping(funding['data_source'])
            funding['data_source'] = DataSourceConfig(resolve(cast(str, source['method'])), dict(_mapping(source.get('params'))))
        kwargs['funding'] = cast(Callable[..., object], FundingConfig)(**funding)
    _ = cast(Callable[..., object], manifest.set_backtest_config)(**kwargs)
